import torch
import torch.nn as nn
import numpy as np
import cv2
import logging
import mediapipe as mp
from pathlib import Path
import os
from ultralytics import YOLO
from train_model import ShootingPoseModel

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def detect_basketball_yolo(frame, ball_model, device):
    """YOLO 모델을 사용하여 프레임에서 농구공을 검출"""
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

class BasketballShootingAnalyzer:
    def __init__(self, model_path='pretrain_model_epoch_10.pth', handedness='right'):
        self.handedness = handedness
        
        self.ball_model = YOLO('best_2.pt')
        self.ball_model.to(device)
        
        self.model = ShootingPoseModel()
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()

    def transform_to_right_handed(self, features):
        transformed = features.copy()
        left_right_pairs = [
            (11, 12),  # 어깨
            (13, 14),  # 팔꿈치
            (15, 16),  # 손목
            (23, 24),  # 엉덩이
            (25, 26),  # 무릎
            (27, 28)   # 발목
        ]
        
        for left_idx, right_idx in left_right_pairs:
            left_start = left_idx * 4
            right_start = right_idx * 4
            temp = transformed[:, left_start:left_start+4].copy()
            transformed[:, left_start:left_start+4] = transformed[:, right_start:right_start+4]
            transformed[:, right_start:right_start+4] = temp
            
        return transformed
    
    def generate_feedback(self, differences):
        """차이점 분석을 바탕으로 피드백 생성"""
        joint_differences = {}
        
        key_joints = {
            'right_shoulder': 12,
            'right_elbow': 14,
            'right_wrist': 16,
            'right_hip': 24,
            'right_knee': 26,
            'right_ankle': 28,
            'left_shoulder': 11,
            'left_elbow': 13,
            'left_wrist': 15,
            'left_hip': 23,
            'left_knee': 25,
            'left_ankle': 27
        }
        
        valid_differences = []
        for joint_name, idx in key_joints.items():
            start_idx = idx * 4
            joint_diff = np.nanmean(differences[start_idx:start_idx+3])  # NaN 값을 무시하고 평균 계산
            if not np.isnan(joint_diff):  # 유효한 차이값만 저장
                joint_differences[joint_name] = joint_diff
                valid_differences.append(joint_diff)
        
        if not valid_differences:  # 유효한 차이값이 없는 경우
            print("Warning: No valid differences found")
            return None
        
        feedback = {
            'major_differences': [],
            'detailed_analysis': {}
        }
        
        # 유효한 차이값들의 평균과 표준편차로 임계값 설정
        threshold = np.mean(valid_differences) + np.std(valid_differences)
        
        for joint, diff in joint_differences.items():
            if diff > threshold:
                feedback['major_differences'].append(joint)
            feedback['detailed_analysis'][joint] = {
                'difference': float(diff),
                'feedback': self._generate_joint_feedback(joint, diff, threshold)
            }
        
        return feedback

    def _generate_joint_feedback(self, joint, difference, threshold):
        if difference <= threshold:
            return f"{joint.replace('_', ' ').title()}의 움직임이 프로 선수와 유사합니다."
        
        feedback_templates = {
            'shoulder': "어깨의 높이와 회전이 프로 선수와 차이가 있습니다. 릴리즈 시 어깨 정렬에 주의해주세요.",
            'elbow': "팔꿈치 각도가 프로 선수와 다릅니다. 더 부드러운 팔꿈치 움직임이 필요합니다.",
            'wrist': "손목 스냅이 프로 선수와 차이가 있습니다. 릴리즈 시 손목 스냅을 더 자연스럽게 해주세요.",
            'hip': "힙의 위치가 프로 선수와 다릅니다. 더 안정적인 자세가 필요합니다.",
            'knee': "무릭 굽힘이 프로 선수와 차이가 있습니다. 점프 시 무릎 사용을 개선해주세요.",
            'ankle': "발목의 움직임이 프로 선수와 다릅니다. 더 안정적인 지지가 필요합니다."
        }
        
        for key, template in feedback_templates.items():
            if key in joint:
                return template
        
        return f"{joint.replace('_', ' ').title()}의 움직임에 개선이 필요합니다."

    def analyze_shooting_mechanics(self, user_sequence):
        print("\nStarting shooting mechanics analysis...")
        
        pose_detector = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        features = []
        frames_per_shot = 30
        
        print("Extracting pose features...")
        for frame in user_sequence:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                features.append(landmarks)
        
        if not features:
            print("No pose features detected")
            return None
        
        print(f"Extracted features from {len(features)} frames")
        features = np.array(features, dtype=np.float32)
        
        if len(features) > frames_per_shot:
            indices = np.linspace(0, len(features)-1, frames_per_shot, dtype=int)
            features = features[indices]
        elif len(features) < frames_per_shot:
            padding = np.tile(features[-1], (frames_per_shot - len(features), 1))
            features = np.vstack((features, padding))
        
        if self.handedness == 'left':
            features = self.transform_to_right_handed(features)
        
        print("Running model inference...")
        with torch.no_grad():
            user_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
            reconstructed = self.model(user_tensor)
            
            # NaN 값 처리
            reconstructed = torch.nan_to_num(reconstructed, nan=0.0)
            user_tensor = torch.nan_to_num(user_tensor, nan=0.0)
            
            print("Generating overlay video...")
            overlay_and_save_video(user_sequence, reconstructed.cpu().numpy()[0], features)
            
            # 차이 계산 및 NaN 처리
            differences = torch.abs(reconstructed - user_tensor)
            differences = differences.cpu().numpy()
            differences = np.nan_to_num(differences, nan=0.0)  # NaN을 0으로 변환
        
        feedback = self.generate_feedback(differences[0])
        if feedback is None:
            return None
            
        similarity = 1 - np.mean(differences[~np.isnan(differences)])  # NaN이 아닌 값들의 평균
        feedback['overall_similarity'] = float(similarity) * 100
        
        if self.handedness == 'left' and feedback is not None:
            feedback = self.swap_feedback_labels(feedback)
        
        print("Analysis completed")
        return feedback

def analyze_video(video_path, model_path='pretrain_model_epoch_10.pth', handedness='right'):
    if not os.path.exists(video_path):
        print(f"Error: Video file does not exist: {video_path}")
        return None
        
    print(f"Opening video file: {video_path}")
    analyzer = BasketballShootingAnalyzer(model_path, handedness=handedness)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        print(f"OpenCV version: {cv2.__version__}")
        print(f"Video file absolute path: {os.path.abspath(video_path)}")
        return None
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video info - Width: {width}, Height: {height}, FPS: {fps}, Total frames: {total_frames}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    if not frames:
        print("No frames were read from the video")
        return None
        
    print(f"Successfully read {len(frames)} frames from video")
    feedback = analyzer.analyze_shooting_mechanics(frames)
    return feedback

def main():
    video_path = "input_video.mp4"
    feedback = analyze_video(video_path)

    if feedback is None:
        print("\nAnalysis failed: No pose detected in the video.")
        return

    print("\nAnalysis Results:")
    print(f"Overall similarity with pro player: {feedback['overall_similarity']:.2f}%")
    print("\nMajor differences found in:", feedback['major_differences'])
    print("\nDetailed Analysis:")
    for joint, analysis in feedback['detailed_analysis'].items():
        print(f"\n{joint}:")
        print(f"Difference: {analysis['difference']:.4f}")
        print(f"Feedback: {analysis['feedback']}")

if __name__ == "__main__":
    main()