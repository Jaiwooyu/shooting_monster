import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import os
import json
from tqdm import tqdm

from .dataset import ShootingDataset
from .model import BasketballShootingAnalyzer
from .utils import calculate_metrics

def train(config):
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터 로드
    train_dataset = ShootingDataset(config['train_data_dir'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    # 모델 초기화
    model = BasketballShootingAnalyzer().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    
    # 학습 로그 초기화
    log_dir = os.path.join(config['log_dir'], datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    
    best_loss = float('inf')
    
    # 학습 루프
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}") as pbar:
            for batch in pbar:
                optimizer.zero_grad()
                
                inputs = batch['input'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, batch['target'].to(device))
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(train_loader)
        
        # 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(log_dir, 'best_model.pth'))
        
        # 로그 저장
        with open(os.path.join(log_dir, 'training_log.json'), 'a') as f:
            log = {
                'epoch': epoch,
                'loss': avg_loss,
                'timestamp': datetime.now().isoformat()
            }
            json.dump(log, f)
            f.write('\n')

if __name__ == "__main__":
    config = {
        'train_data_dir': 'path/to/processed_shots',
        'batch_size': 32,
        'num_workers': 4,
        'learning_rate': 1e-4,
        'epochs': 200,
        'log_dir': 'logs'
    }
    
    train(config)