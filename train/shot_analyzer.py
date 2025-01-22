import torch
import torch.nn as nn
import numpy as np
import cv2
import logging
from pathlib import Path
import os
from ultralytics import YOLO
from train_model import ShootingPoseModel  # 모델 클래스 임포트

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BasketballShootingAnalyzer:
    def __init__(self, model_path='best_pretrain_model.pth', handedness='right'):
        self.handedness = handedness
        self.frame_width = None  # 첫 번째 프레임에서 설정

        # YOLOv8l 일반 객체 탐지 모델 초기화
        self.pose_model = YOLO('yolov8l.pt')
        self.pose_model.to(device)

        # 사람 탐지를 위한 동일한 모델 사용
        self.person_model = YOLO('yolov8l.pt')
        self.person_model.to(device)

        self.model = ShootingPoseModel()
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def extract_pose_features(self, frame):
        results = self.pose_model.predict(frame, imgsz=640, conf=0.4, verbose=False)
        if not results or len(results) == 0:
            logging.error("No detection results from YOLOv8.")
            return None

        res = results[0]
        if not hasattr(res, 'keypoints') or res.keypoints is None:
            logging.error("No keypoints detected in YOLOv8 results.")
            return None

        # 랜드마크 디버깅
        logging.info(f"Detected keypoints: {res.keypoints.shape}")
        keypoints = res.keypoints.cpu().numpy() if torch.cuda.is_available() else res.keypoints.numpy()

        # Convert to 132-dimension format
        landmarks = []
        for kp in keypoints:
            x, y, conf = kp
            landmarks.extend([x, y, 0.0, conf])  # z=0.0 추가

        logging.info(f"Extracted landmarks: {len(landmarks)}")
        return np.array(landmarks)



    def transform_to_right_handed(self, features):
        """
        왼손잡이 사용자의 데이터를 오른손잡이 기준으로 변환.
        여기서는 바운딩 박스 좌표를 좌우 반전.
        """
        # 일반 객체 탐지에서는 handedness 변환을 위해 바운딩 박스 좌우 반전 수행
        transformed = features.copy()
        if self.frame_width is None:
            return transformed

        # 각 프레임에 대해 바운딩 박스 좌표 반전
        for i in range(transformed.shape[0]):
            x1, y1, x2, y2 = transformed[i]
            transformed[i, 0] = self.frame_width - x2
            transformed[i, 2] = self.frame_width - x1
            # y1, y2는 세로 좌표로 반전하지 않음
        return transformed

    def swap_feedback_labels(self, feedback):
        swap_map = {
            'right_shoulder': 'left_shoulder',
            'right_elbow': 'left_elbow',
            'right_wrist': 'left_wrist',
            'right_hip': 'left_hip',
            'right_knee': 'left_knee',
            'right_ankle': 'left_ankle',
            'left_shoulder': 'right_shoulder',
            'left_elbow': 'right_elbow',
            'left_wrist': 'right_wrist',
            'left_hip': 'right_hip',
            'left_knee': 'right_knee',
            'left_ankle': 'right_ankle'
        }

        if 'major_differences' in feedback:
            feedback['major_differences'] = [
                swap_map.get(joint, joint) for joint in feedback['major_differences']
            ]

        if 'detailed_analysis' in feedback:
            new_detailed = {}
            for joint, analysis in feedback['detailed_analysis'].items():
                new_joint = swap_map.get(joint, joint)
                new_detailed[new_joint] = analysis
            feedback['detailed_analysis'] = new_detailed

        return feedback
    
    def analyze_shooting_mechanics(self, user_sequence):
        features = []
        for frame in user_sequence:
            pose_features = self.extract_pose_features(frame)
            if pose_features is not None:
                features.append(pose_features)

        if not features:
            return None

        features = np.array(features)

        # 왼손잡이일 경우 데이터를 오른손잡이 기준으로 변환
        if self.handedness == 'left':
            features = self.transform_to_right_handed(features)

        # 모델 입력을 위한 33 프레임 132차원 벡터로 변환
        # features shape: (num_frames, 4)
        required_frames = 33
        current_frames = features.shape[0]

        # 프레임 수가 33 미만이면 패딩
        if current_frames < required_frames:
            pad = np.zeros((required_frames - current_frames, features.shape[1]))
            features = np.vstack([features, pad])
        
        # 처음 33 프레임 선택
        features = features[:required_frames]
        
        # 33개의 4차원 벡터를 132차원 벡터로 평탄화
        ffeatures = features.flatten()  # shape: (132,)

        with torch.no_grad():
    # features shape 확인 (33, 132)이어야 함)
            assert features.shape == (33, 132), f"Unexpected features shape: {features.shape}"

            # 배치 차원을 추가하여 (1, 33, 132)로 변환
            user_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)  # shape: (1, 33, 132)
            reconstructed = self.model(user_tensor)  # 모델 입력
            differences = torch.abs(reconstructed - user_tensor)
            differences = differences.cpu().numpy()

        feedback = self.generate_feedback(differences[0])

        if self.handedness == 'left' and feedback is not None:
            feedback = self.swap_feedback_labels(feedback)

        return feedback

    def generate_feedback(self, differences):
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
            joint_diff = np.mean(differences[start_idx:start_idx+3])
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

def analyze_video(video_path, model_path='best_pretrain_model.pth'):
    analyzer = BasketballShootingAnalyzer(model_path, handedness='right')

    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return None

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1

    cap.release()

    logging.info(f"Processed {frame_count} frames from video")

    if frame_count == 0:
        logging.error("No frames were read from the video")
        return None

    feedback = analyzer.analyze_shooting_mechanics(frames)

    if feedback is None:
        logging.error("Failed to analyze shooting mechanics")
    else:
        logging.info("Successfully analyzed shooting mechanics")

    return feedback

def main():
    video_path = "../dataset1/input/input_video.mp4"
    feedback = analyze_video(video_path)

    if feedback is None:
        print("\nAnalysis failed: No pose detected in the video.")
        return

    print("\nAnalysis Results:")
    print("Major differences found in:", feedback['major_differences'])
    print("\nDetailed Analysis:")
    for joint, analysis in feedback['detailed_analysis'].items():
        print(f"\n{joint}:")
        print(f"Difference: {analysis['difference']:.4f}")
        print(f"Feedback: {analysis['feedback']}")

if __name__ == "__main__":
    main()
