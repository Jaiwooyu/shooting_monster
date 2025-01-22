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
    def __init__(self, model_path='pretrained_model.pth', handedness='right'):
        # handedness: 'right' 또는 'left'
        self.handedness = handedness
        self.frame_width = None  # 첫 번째 프레임에서 설정


        # 사람 탐지를 위한 모델 초기화 (필요시 사용)
        self.pose_model = YOLO('yolov8l.pt')
        self.pose_model.to(device)

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
        """YOLOv8l-pose를 사용하여 프레임에서 첫 번째 사람에 대한 포즈 특징 추출"""
        if self.frame_width is None:
            self.frame_width = frame.shape[1]

        results = self.pose_model.predict(frame, imgsz=640, conf=0.4, verbose=False)
        if not results or len(results) == 0:
            return None

        res = results[0]
        if res.keypoints is None:
            return None

        try:
            kp = res.keypoints
            raw = kp.data.cpu().numpy() if torch.cuda.is_available() else kp.data.numpy()
        except Exception as e:
            logging.error(f"Error extracting keypoints: {e}")
            return None

        if raw.ndim == 3:
            keypoints = raw[0]
        elif raw.ndim == 2:
            keypoints = raw
        else:
            logging.error(f"Unexpected keypoints shape: {raw.shape}")
            return None

        landmarks = []
        for (x, y, conf) in keypoints:
            # handedness에 따른 좌우 반전은 여기서 하지 않고, 전체 데이터 변환 함수에서 처리
            landmarks.extend([x, y, 0.0, float(conf)])

        if len(landmarks) < 132:
            landmarks.extend([0.0] * (132 - len(landmarks)))

        return np.array(landmarks)

    def transform_to_right_handed(self, features):
        """
        왼손잡이 사용자의 포즈 데이터를 오른손잡이 기준으로 변환.
        여기서는 주요 관절 쌍만 좌우 교환.
        """
        transformed = features.copy()
        # 주요 관절 인덱스 쌍 (왼쪽과 오른쪽 교환)
        left_right_pairs = [
            (11, 12), (13, 14), (15, 16),  # 어깨, 팔꿈치, 손목
            (23, 24), (25, 26), (27, 28)   # 엉덩이, 무릎, 발목
        ]
        for left_idx, right_idx in left_right_pairs:
            left_start = left_idx * 4
            right_start = right_idx * 4
            # 각 프레임에 대해 좌우 교환
            temp = transformed[:, left_start:left_start+4].copy()
            transformed[:, left_start:left_start+4] = transformed[:, right_start:right_start+4]
            transformed[:, right_start:right_start+4] = temp
        return transformed

    def swap_feedback_labels(self, feedback):
        """
        피드백의 관절 라벨을 좌우 반전.
        """
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

        # major_differences 라벨 교환
        if 'major_differences' in feedback:
            feedback['major_differences'] = [
                swap_map.get(joint, joint) for joint in feedback['major_differences']
            ]

        # detailed_analysis 키 교환
        if 'detailed_analysis' in feedback:
            new_detailed = {}
            for joint, analysis in feedback['detailed_analysis'].items():
                new_joint = swap_map.get(joint, joint)
                new_detailed[new_joint] = analysis
            feedback['detailed_analysis'] = new_detailed

        return feedback

    def analyze_shooting_mechanics(self, user_sequence):
        """사용자의 슈팅 메카닉스 분석"""
        features = []
        for frame in user_sequence:
            pose_features = self.extract_pose_features(frame)
            if pose_features is not None:
                features.append(pose_features)

        if not features:
            return None
        
        print(features)

        
        features = np.array(features)

        # 왼손잡이일 경우 데이터를 오른손잡이 기준으로 변환
        if self.handedness == 'left':
            features = self.transform_to_right_handed(features)

        with torch.no_grad():
            user_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
            reconstructed = self.model(user_tensor)
            differences = torch.abs(reconstructed - user_tensor)
            differences = differences.cpu().numpy()

        feedback = self.generate_feedback(differences[0])

        # handedness가 left이면 피드백 라벨을 다시 좌우 반전하여 왼손 기준으로 변환
        if self.handedness == 'left' and feedback is not None:
            feedback = self.swap_feedback_labels(feedback)


        return feedback

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
    analyzer = BasketballShootingAnalyzer(model_path, handedness='left')  # 필요시 handedness 설정

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
    # 예시: 비디오 분석
    video_path = "../dataset1/input_video.mp4"  # 분석할 사용자 영상
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
