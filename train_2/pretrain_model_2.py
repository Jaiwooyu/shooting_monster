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
from ultralytics import YOLO

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
    def __init__(self, input_size=132):
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
        
        x = x.view(batch_size * seq_length, -1)
        encoded = self.encoder(x)
        encoded = encoded.view(batch_size, seq_length, -1)
        
        lstm_out, _ = self.lstm(encoded)
        
        lstm_out = lstm_out.contiguous().view(batch_size * seq_length, -1)
        decoded = self.decoder(lstm_out)
        decoded = decoded.view(batch_size, seq_length, -1)
        
        return decoded

class ShootingDataset(Dataset):
    def __init__(self, data_dir, ball_model, is_elite_dataset=False, frames_per_shot=30):
        self.data_dir = Path(data_dir)
        self.is_elite_dataset = is_elite_dataset
        self.frames_per_shot = frames_per_shot
        self.ball_model = ball_model

        # MediaPipe Pose 초기화
        self.pose_estimator = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        if is_elite_dataset:
            self.video_files = []
            for player_dir in self.data_dir.glob('player*'):
                self.video_files.extend(list(player_dir.glob('*.mp4')))
                self.video_files.extend(list(player_dir.glob('*.mov')))
        else:
            self.sequences = list(self.data_dir.glob('sequence_*'))
            
        count = len(self.video_files) if is_elite_dataset else len(self.sequences)
        print(f"Found {count} {'videos' if is_elite_dataset else 'sequences'} in {data_dir}")

    def process_frames(self, frames):
        features = []

        for frame_idx, frame in enumerate(frames):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_estimator.process(frame_rgb)
                
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                features.append(landmarks)

        if not features:
            return np.zeros((self.frames_per_shot, 132), dtype=np.float32)
        features = np.array(features, dtype=np.float32)

        if len(features) > self.frames_per_shot:
            indices = np.linspace(0, len(features)-1, self.frames_per_shot, dtype=int)
            features = features[indices]
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

            if not frames:
                return None

            features = self.process_frames(frames)
            print(f"Processing item: {idx}", end='\r')
            
            if np.all(features == 0):
                return None
                
            return torch.from_numpy(features)
                
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            return None

def train_model(model, train_loader, epochs=80, learning_rate=0.001, is_pretrain=True):
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
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            ncols=100,
            leave=True
        )
        
        for batch_features in progress_bar:
            try:
                batch_features = batch_features.to(device)
                optimizer.zero_grad()
                reconstructed = model(batch_features)
                loss = criterion(reconstructed, batch_features)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                continue
        
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        print(f"\nEpoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        history['epoch'].append(epoch + 1)
        history['loss'].append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'best_{model_prefix}_model.pth')
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'{model_prefix}_model_epoch_{epoch+1}.pth')
    
    return model, history

def main(pretrain=True, finetune=True):
    ball_model = YOLO('best_2.pt')
    ball_model.to(device)
    
    try:
        if pretrain:
            print("\nStarting pre-training...")
            pretrain_dataset = ShootingDataset("elite_players_dataset", ball_model, is_elite_dataset=True)
            pretrain_loader = DataLoader(
                pretrain_dataset,
                batch_size=32,
                shuffle=True,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
                collate_fn=lambda x: torch.stack([item for item in x if item is not None])
            )
            
            model = ShootingPoseModel().to(device)
            
            pretrain_model, pretrain_history = train_model(
                model, 
                pretrain_loader,
                epochs=100,
                learning_rate=0.0005,
                is_pretrain=True
            )
            
            torch.save(pretrain_model.state_dict(), 'pretrained_model.pth')
            print("Pre-training completed!")

        if finetune:
            print("\nStarting fine-tuning...")
            youtube_dataset = ShootingDataset("basketball_shot_dataset", ball_model, is_elite_dataset=False)
            youtube_loader = DataLoader(
                youtube_dataset,
                batch_size=32,
                shuffle=True,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
                collate_fn=lambda x: torch.stack([item for item in x if item is not None])
            )
            
            fine_tune_model = ShootingPoseModel().to(device)
            if os.path.exists('pretrained_model.pth'):
                fine_tune_model.load_state_dict(torch.load('pretrained_model.pth', map_location=device))
            
            final_model, finetune_history = train_model(
                fine_tune_model,
                youtube_loader,
                epochs=100,
                learning_rate=0.0001,
                is_pretrain=False
            )
            
            torch.save(final_model.state_dict(), 'final_model.pth')
            print("Fine-tuning completed!")
            
    except Exception as e:
        print(f"Error in main: {e}")
        raise

if __name__ == '__main__':
    main(pretrain=True, finetune=True)