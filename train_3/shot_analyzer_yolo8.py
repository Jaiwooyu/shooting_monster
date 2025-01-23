import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
import logging
from pathlib import Path
import os
from train_model import ShootingPoseModel  # 모델 클래스 임포트

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BasketballShootingAnalyzer:
    def __init__(self, model_path='pretrained_model.pth'):
        self.pose_detector = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.model = ShootingPoseModel()
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(device)
        self.model.eval()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def extract_pose_features(self, frame):
        """프레임에서 포즈 특징 추출"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None
            
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            
        return np.array(landmarks)
    
    def analyze_shooting_mechanics(self, user_sequence):
        """사용자의 슈팅 메카닉스 분석"""
        features = []
        for frame in user_sequence:
            pose_features = self.extract_pose_features(frame)
            if pose_features is not None:
                features.append(pose_features)
        
        if not features:
            return None
            
        features = np.array(features)
        
        with torch.no_grad():
            user_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
            reconstructed = self.model(user_tensor)
            differences = torch.abs(reconstructed - user_tensor)
            differences = differences.cpu().numpy()
            
        return self.generate_feedback(differences[0])
    
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
        
        for joint_name, idx in key_joints.items():
            start_idx = idx * 4
            joint_diff = np.mean(differences[:, start_idx:start_idx+3])
            joint_differences[joint_name] = joint_diff
        
        feedback = {
            'major_differences': [],
            'detailed_analysis': {}
        }
        
        threshold = np.mean(list(joint_differences.values())) + np.std(list(joint_differences.values()))
        
        for joint, diff in joint_differences.items():
            if diff > threshold:
                feedback['major_differences'].append(joint)
            feedback['detailed_analysis'][joint] = {
                'difference': float(diff),
                'feedback': self._generate_joint_feedback(joint, diff, threshold)
            }
        
        return feedback
    
    def _generate_joint_feedback(self, joint, difference, threshold):
        """개별 관절에 대한 구체적 피드백 생성"""
        if difference <= threshold:
            return f"{joint.replace('_', ' ').title()}의 움직임이 프로 선수와 유사합니다."
        
        feedback_templates = {
            'shoulder': "어깨의 높이와 회전이 프로 선수와 차이가 있습니다. 릴리즈 시 어깨 정렬에 주의해주세요.",
            'elbow': "팔꿈치 각도가 프로 선수와 다릅니다. 더 부드러운 팔꿈치 움직임이 필요합니다.",
            'wrist': "손목 스냅이 프로 선수와 차이가 있습니다. 릴리즈 시 손목 스냅을 더 자연스럽게 해주세요.",
            'hip': "힙의 위치가 프로 선수와 다릅니다. 더 안정적인 자세가 필요합니다.",
            'knee': "무릎 굽힘이 프로 선수와 차이가 있습니다. 점프 시 무릎 사용을 개선해주세요.",
            'ankle': "발목의 움직임이 프로 선수와 다릅니다. 더 안정적인 지지가 필요합니다."
        }
        
        for key, template in feedback_templates.items():
            if key in joint:
                return template
        
        return f"{joint.replace('_', ' ').title()}의 움직임에 개선이 필요합니다."

def analyze_video(video_path, model_path='pretrained_model.pth'):
    """비디오 파일 분석"""
    analyzer = BasketballShootingAnalyzer(model_path)
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    feedback = analyzer.analyze_shooting_mechanics(frames)
    return feedback

def main():
    # 예시: 비디오 분석
    video_path = "dataset1/input/input_video.mp4"  # 분석할 사용자 영상
    feedback = analyze_video(video_path)
    
    print("\nAnalysis Results:")
    print("Major differences found in:", feedback['major_differences'])
    print("\nDetailed Analysis:")
    for joint, analysis in feedback['detailed_analysis'].items():
        print(f"\n{joint}:")
        print(f"Difference: {analysis['difference']:.4f}")
        print(f"Feedback: {analysis['feedback']}")

if __name__ == "__main__":
    main()