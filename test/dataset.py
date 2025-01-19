import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import random
from PIL import Image
import json

class ShootingDataset(Dataset):
    def __init__(self, data_dir, sequence_length=90):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.samples = self.load_samples()

    def load_samples(self):
        samples = []
        for sequence_dir in os.listdir(self.data_dir):
            sequence_path = os.path.join(self.data_dir, sequence_dir)
            if os.path.isdir(sequence_path):
                samples.append(sequence_path)
        return samples

    def load_sequence(self, sequence_path):
        frames = []
        for frame_name in sorted(os.listdir(sequence_path)):
            if frame_name.endswith('.jpg'):
                frame_path = os.path.join(sequence_path, frame_name)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    frames.append(frame)
        return frames
        
    def prepare_sequence(self, sequence):
        if len(sequence) > self.sequence_length:
            start = random.randint(0, len(sequence) - self.sequence_length)
            sequence = sequence[start:start + self.sequence_length]
        else:
            padding = [sequence[-1]] * (self.sequence_length - len(sequence))
            sequence = sequence + padding

        sequence = np.array(sequence) / 255.0
        sequence = torch.from_numpy(sequence).float()
        sequence = sequence.permute(3, 0, 1, 2)
        
        return {'input': sequence}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence_path = self.samples[idx]
        sequence = self.load_sequence(sequence_path)
        return self.prepare_sequence(sequence)