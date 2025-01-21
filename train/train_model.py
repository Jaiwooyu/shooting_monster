import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
import logging
import os
from tqdm import tqdm
import json
from ultralytics import YOLO  # Ultralytics YOLO 임포트

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def detect_basketball_yolo(frame, ball_model, device):
    """
    YOLO 모델을 사용하여 프레임에서 농구공을 검출.
    """
    ball_results = ball_model.predict(frame, imgsz=1280, conf=0.2, verbose=False, classes=[0])
    ball_det = ball_results[0].boxes

    if torch.cuda.is_available():
        ball_boxes = ball_det.xyxy.cpu().numpy()
    else:
        ball_boxes = ball_det.xyxy.numpy()

    ball_box = None
    max_ball_area = 0

    for j in range(len(ball_boxes)):
        x1, y1, x2, y2 = ball_boxes[j]
        area = (x2 - x1) * (y2 - y1)
        if area > max_ball_area:
            max_ball_area = area
            ball_box = (int(x1), int(y1), int(x2), int(y2))

    return ball_box

def extract_pose_landmarks_yolov8(frame, pose_model):
        """
        YOLOv8 포즈 모델을 사용하여 프레임에서 사람의 포즈 랜드마크 추출.
        반환 형식은 [num_keypoints, 3] (x, y, confidence)
        """
        results = pose_model.predict(frame, imgsz=640, conf=0.5, verbose=False)
        
        # 첫 번째 사람의 포즈 결과 사용 (필요에 따라 다중 사람 처리 가능)
        if results and len(results) > 0:
            result = results[0]
            if result.keypoints is not None and len(result.keypoints) > 0:
                # keypoints는 shape (num_keypoints, 3) : (x, y, confidence)
                keypoints = result.keypoints.cpu().numpy() if torch.cuda.is_available() else result.keypoints.numpy()
                return keypoints
        return None

class ShootingPoseModel(nn.Module):
    def __init__(self, input_size=132):  # 33개 랜드마크 * 4 (x, y, z, visibility)
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )
        
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, input_size)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # Reshape for encoder
        x = x.view(batch_size * seq_length, -1)
        encoded = self.encoder(x)
        encoded = encoded.view(batch_size, seq_length, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(encoded)
        
        # Reshape for decoder
        lstm_out = lstm_out.contiguous().view(batch_size * seq_length, -1)
        decoded = self.decoder(lstm_out)
        decoded = decoded.view(batch_size, seq_length, -1)
        
        return decoded

class ShootingDataset(Dataset):
    def __init__(self, data_dir, ball_model, is_elite_dataset=False, frames_per_shot=30):
        self.data_dir = Path(data_dir)
        self.is_elite_dataset = is_elite_dataset
        self.frames_per_shot = frames_per_shot
        self.ball_model = ball_model  # YOLO 농구공 탐지 모델

        if is_elite_dataset:
            self.video_files = []
            for player_dir in self.data_dir.glob('player*'):
                self.video_files.extend(list(player_dir.glob('*.mp4')))
                self.video_files.extend(list(player_dir.glob('*.mov')))
        else:
            self.sequences = list(self.data_dir.glob('sequence_*'))
            
        count = len(self.video_files) if is_elite_dataset else len(self.sequences)
        dataset_type = 'videos' if is_elite_dataset else 'sequences'
        logging.info(f"Found {count} {dataset_type} in {data_dir}")

    def get_pose_detector(self):
        """각 프로세스마다 새로운 MediaPipe Pose 객체 생성"""
        return mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )

    def process_frames(self, frames):
        features = []
        # YOLOv8l-pose 모델 초기화
        pose_model = YOLO('yolov8l-pose.pt')
        pose_model.to(device)
        ball_model = self.ball_model

        pose_buffer = []         # 농구공 인식 실패 시 버퍼링할 프레임별 사람들의 포즈 데이터
        ball_detected = False    # 농구공 최초 인식 여부 플래그

        for frame in frames:
            # YOLOv8l-pose를 사용하여 프레임 내 모든 사람의 포즈 추정
            results = pose_model.predict(frame, imgsz=640, conf=0.5, verbose=False)

            # 모든 인물의 포즈 데이터를 추출
            persons = []
            for res in results:
                if res.keypoints is not None and len(res.keypoints) > 0:
                    keypoints = res.keypoints.cpu().numpy() if torch.cuda.is_available() else res.keypoints.numpy()
                    # 각 인물의 관절 데이터를 (x, y, confidence) 형태로 추출
                    # COCO 기준 17개 관절 예상
                    landmarks = []
                    for (x, y, conf) in keypoints:
                        landmarks.extend([x, y, 0.0, float(conf)])  # z=0.0, visibility=conf
                    # 33개 랜드마크 필요 시 패딩 (여기서는 17개 기준 처리)
                    if len(landmarks) < 132:
                        landmarks.extend([0.0] * (132 - len(landmarks)))
                    persons.append(landmarks)

            # YOLO를 사용하여 농구공 검출
            ball_box = detect_basketball_yolo(frame, ball_model, device)

            # 농구공이 아직 인식되지 않은 경우
            if not ball_detected:
                if ball_box is None:
                    # 농구공을 인식하지 못했으므로, 현재 프레임의 모든 사람 데이터 버퍼에 저장
                    pose_buffer.append(persons)
                    continue
                else:
                    # 최초 농구공 인식 시점
                    ball_detected = True

                    # 농구공 중심 좌표 계산
                    bx1, by1, bx2, by2 = ball_box
                    center_x = (bx1 + bx2) // 2
                    center_y = (by1 + by2) // 2

                    # 현재 프레임에서 농구공과 가까운 손을 가진 사람 찾기
                    min_distance = float('inf')
                    person_holding_ball = None
                    hand_landmark_indices = [15, 19]  # MediaPipe Pose 기준 오른손(15), 왼손(19) 예시
                                                    # YOLOv8l-pose 모델의 경우 관절 인덱스는 COCO 기준
                                                    # 오른손, 왼손에 해당하는 인덱스 확인 필요

                    # 각 인물에 대해 손 위치와 농구공 사이 거리 계산
                    for person in persons:
                        for idx in hand_landmark_indices:
                            start = idx * 4
                            if start + 1 < len(person):
                                x = person[start]
                                y = person[start + 1]
                                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                                if distance < min_distance:
                                    min_distance = distance
                                    person_holding_ball = person

                    # 손과 농구공 사이 거리 임계값 이하 확인
                    hand_distance_threshold = 50  # 필요에 따라 조정
                    if min_distance < hand_distance_threshold and person_holding_ball is not None:
                        # 버퍼에 저장된 이전 모든 프레임에서 해당 사람의 데이터 수집
                        for frame_persons in pose_buffer:
                            # 단순화를 위해 각 프레임의 첫 번째 사람 데이터 사용
                            # 실제 구현에서는 동일 인물 식별 필요
                            if frame_persons and len(frame_persons) > 0:
                                features.append(frame_persons[0])
                        # 현재 프레임의 해당 사람 데이터 추가
                        features.append(person_holding_ball)
                    # 농구공 인식 후 추가 처리 생략
                    break  # 최초 인식 시점에서 수집 완료 후 루프 종료

            # 농구공이 이미 인식된 이후의 프레임 처리는 생략 또는 별도 구현

        # 모델 및 리소스 정리
        pose_model = None
        torch.cuda.empty_cache()

        # features가 비어있으면 제로 배열 반환
        if not features:
            return np.zeros((self.frames_per_shot, 132), dtype=np.float32)

        features = np.array(features, dtype=np.float32)

        # frames_per_shot 크기에 맞추기
        if len(features) > self.frames_per_shot:
            features = features[:self.frames_per_shot]
        elif len(features) < self.frames_per_shot:
            padding = np.tile(features[-1], (self.frames_per_shot - len(features), 1))
            features = np.vstack((features, padding))

        return features

    

    def extract_frames_from_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return frames
        
        frame_indices = np.linspace(0, total_frames-1, self.frames_per_shot, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames

    def __len__(self):
        return len(self.video_files) if self.is_elite_dataset else len(self.sequences)
    
    def __getitem__(self, idx):
        try:
            if self.is_elite_dataset:
                frames = self.extract_frames_from_video(self.video_files[idx])
            else:
                seq_path = self.sequences[idx]
                frames = []
                for frame_file in sorted(seq_path.glob('frame_*.jpg')):
                    frame = cv2.imread(str(frame_file))
                    if frame is not None:
                        frames.append(frame)
            
            features = self.process_frames(frames)
            return torch.from_numpy(features)
            
        except Exception as e:
            logging.error(f"Error processing item {idx}: {e}")
            return torch.zeros((self.frames_per_shot, 132), dtype=torch.float32)

def train_model(model, train_loader, epochs=100, learning_rate=0.001, is_pretrain=True):
    """모델 학습"""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model = model.to(device)
    
    model_prefix = "pretrain" if is_pretrain else "finetuned"
    best_loss = float('inf')
    
    history = {
        'epoch': [],
        'loss': []
    }
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_features in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            try:
                optimizer.zero_grad()
                
                # DataLoader에서 반환된 배치가 리스트인 경우 스택으로 변환
                if isinstance(batch_features, list):
                    batch_features = torch.stack(batch_features)
                
                batch_features = batch_features.to(device)
                
                reconstructed = model(batch_features)
                loss = criterion(reconstructed, batch_features)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
            except Exception as e:
                logging.error(f"Error in batch: {e}")
                continue
        
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        
        logging.info(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")
        
        history['epoch'].append(epoch + 1)
        history['loss'].append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            # DataParallel 사용하지 않을 경우 바로 저장
            torch.save(model.state_dict(), f'best_{model_prefix}_model.pth')
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'{model_prefix}_model_epoch_{epoch+1}.pth')
            with open(f'{model_prefix}_history.json', 'w') as f:
                json.dump(history, f)
    
    return model, history

def main(pretrain=True, finetune=True):
    # YOLO 농구공 탐지 모델 초기화
    ball_model = YOLO('best_2.pt')
    ball_model.to(device)
    
    try:
        if pretrain and not os.path.exists("elite_players_dataset"):
            logging.error("Error: elite_players_dataset directory not found")
            return
            
        if finetune and not os.path.exists("basketball_shot_dataset"):
            logging.error("Error: basketball_shot_dataset directory not found")
            return
        
        if pretrain:
            logging.info("Starting pre-training with elite players dataset...")
            pretrain_dataset = ShootingDataset("elite_players_dataset", ball_model, is_elite_dataset=True)
            pretrain_loader = DataLoader(
                pretrain_dataset,
                batch_size=16,
                shuffle=True,
                num_workers=0,            # 멀티프로세싱 비활성화
                pin_memory=torch.cuda.is_available()
            )
            
            model = ShootingPoseModel().to(device)
            
            pretrain_model, pretrain_history = train_model(
                model, 
                pretrain_loader,
                epochs=30,
                learning_rate=0.001,
                is_pretrain=True
            )
            
            torch.save(pretrain_model.state_dict(), 'pretrained_model.pth')
            logging.info("Pre-training completed!")

        if finetune:
            logging.info("Starting fine-tuning with YouTube dataset...")
            youtube_dataset = ShootingDataset("basketball_shot_dataset", ball_model, is_elite_dataset=False)
            youtube_loader = DataLoader(
                youtube_dataset,
                batch_size=8,
                shuffle=True,
                num_workers=0,            # 멀티프로세싱 비활성화
                pin_memory=torch.cuda.is_available()
            )
            
            fine_tune_model = ShootingPoseModel().to(device)
            if os.path.exists('pretrained_model.pth'):
                fine_tune_model.load_state_dict(torch.load('pretrained_model.pth', map_location=device))
            
            final_model, finetune_history = train_model(
                fine_tune_model,
                youtube_loader,
                epochs=50,
                learning_rate=0.0001,
                is_pretrain=False
            )
            
            torch.save(final_model.state_dict(), 'final_model.pth')
            logging.info("Fine-tuning completed!")
            
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == '__main__':
    main(pretrain=True, finetune=True)
