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
    def __init__(self, data_dir, is_elite_dataset=False, frames_per_shot=30):
        self.data_dir = Path(data_dir)
        self.is_elite_dataset = is_elite_dataset
        self.frames_per_shot = frames_per_shot
        
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
        """각 프로세스마다 새로운 MediaPipe 객체 생성"""
        return mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )

    def process_frames(self, frames):
        features = []
        pose_detector = self.get_pose_detector()
        
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                features.append(landmarks)
        
        pose_detector.close()
        
        # 랜드마크 검출 실패 시
        if not features:
            return np.zeros((self.frames_per_shot, 132), dtype=np.float32)
        
        features = np.array(features, dtype=np.float32)
        
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
    try:
        if pretrain and not os.path.exists("elite_players_dataset"):
            logging.error("Error: elite_players_dataset directory not found")
            return
            
        if finetune and not os.path.exists("basketball_shot_dataset"):
            logging.error("Error: basketball_shot_dataset directory not found")
            return
        
        if pretrain:
            logging.info("Starting pre-training with elite players dataset...")
            pretrain_dataset = ShootingDataset("elite_players_dataset", is_elite_dataset=True)
            pretrain_loader = DataLoader(
                pretrain_dataset,
                batch_size=16,
                shuffle=True,
                num_workers=0,            # 멀티프로세싱 비활성화
                pin_memory=torch.cuda.is_available()
            )
            
            model = ShootingPoseModel()
            # DataParallel 사용하지 않고 단일 GPU 사용
            model = model.to(device)
            
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
            youtube_dataset = ShootingDataset("basketball_shot_dataset", is_elite_dataset=False)
            youtube_loader = DataLoader(
                youtube_dataset,
                batch_size=8,
                shuffle=True,
                num_workers=0,            # 멀티프로세싱 비활성화
                pin_memory=torch.cuda.is_available()
            )
            
            fine_tune_model = ShootingPoseModel()
            fine_tune_model = fine_tune_model.to(device)
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
