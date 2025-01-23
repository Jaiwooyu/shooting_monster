#!/usr/bin/env python3
# shot_analyzer.py

import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import cv2
import logging
from pathlib import Path
import os
import json
import sys
import math  # 추가: NaN 체크를 위해 math 모듈 임포트

from ultralytics import YOLO
from train_model import ShootingPoseModel

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", file=sys.stderr)    

def filter_normalized_landmarks(input_landmarks, normalized_landmarks, threshold=0.2):
   """정규화된 랜드마크와 입력 랜드마크의 차이가 임계값을 넘으면 입력값 사용"""
   filtered = normalized_landmarks.copy()
   
   for frame_idx in range(len(input_landmarks)):
       input_frame = input_landmarks[frame_idx].reshape(-1, 4)
       norm_frame = filtered[frame_idx].reshape(-1, 4)
       
       for joint_idx in range(33):
           coord_diff = np.abs(input_frame[joint_idx, :3] - norm_frame[joint_idx, :3])
           if np.max(coord_diff) > threshold:
               filtered[frame_idx].reshape(-1, 4)[joint_idx] = input_frame[joint_idx]
               
   return filtered

def normalize_reconstructed_landmarks(input_landmarks, reconstructed_landmarks):
    """
    reconstructed 랜드마크를 입력 랜드마크의 위치와 스케일에 맞게 정규화
    """
    normalized = reconstructed_landmarks.copy()
    
    for frame_idx in range(len(input_landmarks)):
        # [33, 4] 형태로 reshape (x, y, z, visibility)
        input_frame = input_landmarks[frame_idx].reshape(-1, 4)
        recon_frame = normalized[frame_idx].reshape(-1, 4)
        
        # x,y 좌표의 중심점 계산
        input_centroid = np.mean(input_frame[:, :2], axis=0)
        recon_centroid = np.mean(recon_frame[:, :2], axis=0)
        
        # 스케일 계산 (x,y 좌표의 표준편차 비율)
        input_scale = np.std(input_frame[:, :2])
        recon_scale = np.std(recon_frame[:, :2])
        scale_ratio = input_scale / recon_scale if recon_scale > 0 else 1.0
        
        # 중심점 이동 및 스케일 조정 (x,y,z 좌표만)
        normalized[frame_idx].reshape(-1, 4)[:, :3] = (
            (recon_frame[:, :3] - np.append(recon_centroid, 0)) * scale_ratio + 
            np.append(input_centroid, 0)
        )
    
    filtered = filter_normalized_landmarks(input_landmarks, normalized)
    
    return filtered

def overlay_and_save_video(user_sequence, reconstructed_landmarks, original_landmarks, output_path, fps, mix_ratio=0.2):
    """
    비디오 시퀀스에 원본, 재구성, 혼합 랜드마크를 각각 다른 색상으로 오버레이하고 저장
    """
    print("\nStarting video overlay process...", file=sys.stderr)
    
    mp_pose = mp.solutions.pose

    def draw_landmarks_manual(image, landmarks_array, color):
        landmarks = {}
        for idx in range(33):  # MediaPipe는 33개의 랜드마크 사용
            start_idx = idx * 4
            x = landmarks_array[start_idx]
            y = landmarks_array[start_idx + 1]
            visibility = landmarks_array[start_idx + 3]
            
            # 좌표가 0~1 사이인지 판단하여 변환
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                pixel_x = int(x * image.shape[1])
                pixel_y = int(y * image.shape[0])
            else:
                pixel_x = int(x)
                pixel_y = int(y)
            
            pixel_x = max(0, min(pixel_x, image.shape[1]-1))
            pixel_y = max(0, min(pixel_y, image.shape[0]-1))
            
            if visibility > 0.5:
                landmarks[idx] = (pixel_x, pixel_y)
        
        # 랜드마크 연결 선과 점 그리기
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if (start_idx in landmarks) and (end_idx in landmarks):
                cv2.line(image, landmarks[start_idx], landmarks[end_idx], color, 2)
        for point in landmarks.values():
            cv2.circle(image, point, 4, color, -1)

    if len(user_sequence) == 0:
        raise ValueError("Empty video sequence")
        
    height, width = user_sequence[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing {len(user_sequence)} frames...", file=sys.stderr)

    for frame_idx in range(len(user_sequence)):
        frame = user_sequence[frame_idx].copy()
        
        if (frame_idx < len(original_landmarks) 
            and frame_idx < len(reconstructed_landmarks)):
            orig = original_landmarks[frame_idx]
            recon = reconstructed_landmarks[frame_idx]
            combined_landmarks = (1 - mix_ratio) * orig + mix_ratio * recon
            
            # 원본 랜드마크: 빨간색
            draw_landmarks_manual(frame, orig, (0, 0, 255))
            # 재구성 랜드마크: 파란색
            draw_landmarks_manual(frame, recon, (255, 0, 0))
            # 혼합 랜드마크: 녹색
            draw_landmarks_manual(frame, combined_landmarks, (0, 255, 0))
            
            cv2.putText(frame, f'Frame: {frame_idx}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, 'Red: Original, Blue: Reconstructed, Green: Mixed', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Video saved to: {output_path}", file=sys.stderr)

class BasketballShootingAnalyzer:
    def __init__(self, model_path='best_pretrain_model_train_1.pth', handedness='right'):
        self.handedness = handedness
        
        # YOLO 모델 (볼 감지) - 여기서는 사용하지 않더라도 예시 유지
        self.ball_model = YOLO('best_2.pt')
        self.ball_model.to(device)
        
        # 포즈 추정 모델 초기화
        self.model = ShootingPoseModel()
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()

    def transform_to_right_handed(self, features):
        """
        왼손잡이 사용자의 포즈 데이터를 오른손잡이 기준으로 변환.
        """
        transformed = features.copy()
        # 주요 관절 인덱스 쌍 (왼쪽과 오른쪽 교환)
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
            # 각 프레임에 대해 좌우 교환
            temp = transformed[:, left_start:left_start+4].copy()
            transformed[:, left_start:left_start+4] = transformed[:, right_start:right_start+4]
            transformed[:, right_start:right_start+4] = temp
            
        return transformed

    def swap_feedback_labels(self, feedback):
        """왼손잡이용 피드백 라벨 변환"""
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
    
    # --------------------------------------------------------------
    # 점수 계산 및 피드백 생성
    # --------------------------------------------------------------

    def analyze_shooting_form(self, features, differences, normalized=None):
        """슈팅 폼 분석 및 점수 책정"""
        
        def calculate_angle(p1, p2, p3):
            """세 점 사이의 각도 계산"""
            a = np.array(p1[:2])  # x,y만 사용
            b = np.array(p2[:2])
            c = np.array(p3[:2])
            
            ba = a - b
            bc = c - b
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            
            return np.degrees(angle)

        scores = {
            'elbow_extension': 0,      # 팔꿈치 펴짐
            'wrist_flexion': 0,        # 손목 꺾임
            'shoulder_alignment': 0,   # 어깨 정렬
            'timing': 0,               # 무릎-팔꿈치 타이밍
            'arm_angle': 0             # 팔 각도
        }
        
        frame_count = len(features)
        max_elbow_angle = 0
        max_wrist_flex = 0
        max_arm_angle = 0
        release_frame = None
        
        # 릴리즈 포인트 찾기 (손목 높이가 최대인 프레임)
        max_wrist_height = 0
        for i in range(frame_count):
            right_wrist = features[i][16*4:16*4+2]  # right_wrist의 x,y
            if right_wrist[1] > max_wrist_height:
                max_wrist_height = right_wrist[1]
                release_frame = i
        
        if release_frame is None:
            return None
        
        # 릴리즈 전후 10프레임 분석
        start_frame = max(0, release_frame - 10)
        end_frame = min(frame_count, release_frame + 10)
        
        for i in range(0, frame_count):
            frame = features[i].reshape(-1, 4)
            
            # 팔꿈치 각도 계산
            shoulder = frame[12][:2]  # right_shoulder
            elbow = frame[14][:2]     # right_elbow
            wrist = frame[16][:2]     # right_wrist
            
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            max_elbow_angle = max(max_elbow_angle, elbow_angle)
            
            # 손목 유연도 계산 (손목 각도)
            # 이상적인 손목 각도 예시: 45도
            ideal_wrist_angle = 60
            # 손목 각도 계산: 팔꿈치, 손목, 손가락 (손가락 데이터가 없으면 어깨, 팔꿈치, 손목으로 대체)
            if len(frame) > 20:
                # MediaPipe 손가락 랜드마크가 추가된 경우
                current_wrist_angle = calculate_angle(frame[14][:2], frame[16][:2], frame[20][:2])
            else:
                current_wrist_angle = calculate_angle(shoulder, elbow, wrist)
            wrist_flexion_diff = abs(ideal_wrist_angle - current_wrist_angle)
            max_wrist_flex = max(max_wrist_flex, wrist_flexion_diff)
            
            # 어깨 정렬 검사
            left_shoulder = frame[11][:2]
            right_shoulder = frame[12][:2]
            shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])  # y축 차이
            
            # 무릎 펴짐 계산
            right_hip = frame[24][:2]
            right_knee = frame[26][:2]
            right_ankle = frame[28][:2]
            knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

            arm_angle = calculate_angle(elbow, right_shoulder, right_hip)
            max_arm_angle = max(max_arm_angle, arm_angle)
            
            # 점수 계산
            if i == release_frame:
                # 팔꿈치 펴짐 (170도 이상이면 만점으로 상향)
                scores['elbow_extension'] = min(100, (max_elbow_angle / 175) * 100)
                
                # 어깨 정렬 (차이가 적을수록 높은 점수, 페널티 상향)
                scores['shoulder_alignment'] = max(0, 100 - (shoulder_diff * 1000))  # 기존 500에서 700으로 증가
                                
                # 손목 유연도 (이상적 차이가 적을수록 높은 점수)
                # 역치를 낮추기 위해 스케일링 조정
                wrist_flexion_score = max(0, 100 - (wrist_flexion_diff * 5))  # 예: 45도 차이 시 10점
                scores['wrist_flexion'] = wrist_flexion_score
        
        # 타이밍 점수 (무릎과 팔꿈치의 최대 펴짐 시점 차이)
        knee_max_frame = start_frame + np.argmax([calculate_angle(features[i].reshape(-1,4)[24][:2],
                                                                features[i].reshape(-1,4)[26][:2],
                                                                features[i].reshape(-1,4)[28][:2])
                                                for i in range(start_frame, end_frame)])
        elbow_max_frame = start_frame + np.argmax([calculate_angle(features[i].reshape(-1,4)[12][:2],
                                                                    features[i].reshape(-1,4)[14][:2],
                                                                    features[i].reshape(-1,4)[16][:2])
                                                    for i in range(start_frame, end_frame)])
        timing_diff = abs(knee_max_frame - elbow_max_frame)
        scores['timing'] = max(0, 100 - (timing_diff * 15))
        scores['arm_angle'] = max(0, 100 - abs(140 - max_arm_angle) * 2)
        
        # differences 데이터 활용하여 특정 관절의 움직임 유사도 반영
        # 양쪽 어깨, 오른팔 손목의 차이 데이터 추출
        specific_joints = [
            'left_shoulder', 'right_shoulder',
            'right_wrist',
            'left_elbow', 'right_elbow',
            # 'right_finger' 추가 가능
        ]
        specific_indices = {
            'left_shoulder': 11 * 4,
            'right_shoulder': 12 * 4,
            'right_wrist': 16 * 4,
            'left_elbow': 13 * 4,      # 52
            'right_elbow': 14 * 4,     # 56
            # 'right_finger': 추가적인 인덱스 필요
        }
        print(f"differences: {differences}", file=sys.stderr)
        differences = differences[0]
        specific_diffs = []
        for joint in specific_joints:
            idx = specific_indices.get(joint, None)
            if idx is not None and idx + 3 <= len(differences):
                diff = differences[idx:idx+3].mean()
                specific_diffs.append(diff)
        print(f"specific_diffs {specific_diffs}", file=sys.stderr)
        print(f"differences shape: {differences.shape}", file=sys.stderr)  # 디버깅: 배열의 형태 출력


        if specific_diffs:
            specific_avg_diff = np.mean(specific_diffs)
            print(f"specific_avg_diff {specific_avg_diff}", file=sys.stderr)
            similarity_specific = max(0, 100 - (specific_avg_diff * 500))  # 스케일 조정
        else:
            similarity_specific = 100  # 특정 관절 데이터가 없을 경우 만점
        
        # 기존 점수들의 가중치 적용 (전체 점수의 50%)
        weights = {
            'elbow_extension': 0.15,
            'wrist_flexion': 0.3,
            'shoulder_alignment': 0.05,
            'timing': 0.2,
            'arm_angle': 0.3
        }
        form_score = sum(scores[k] * weights[k] for k in weights.keys())
        
        # 특정 관절 유사도 점수 (전체 점수의 50%)
        similarity_score = similarity_specific * 0.5  # 50% 가중치
        
        # 최종 점수 계산
        final_score = (form_score * 0.5) + similarity_score
        
        print(f"score {round(final_score, 2)}", file=sys.stderr)
        print(f"detailed_scores { {k: round(v, 2) for k, v in scores.items()} }", file=sys.stderr)
        print(f"similarity_specific {round(similarity_specific, 2)}", file=sys.stderr)
        print(f"similarity_score {round(similarity_score, 2)}", file=sys.stderr)
        print(f"form_score {round(form_score, 2)}", file=sys.stderr)
        return final_score


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
            joint_diff = np.mean(differences[start_idx:start_idx+3])
            joint_differences[joint_name] = joint_diff
        
        feedback = {
            'major_differences': [],
            'detailed_analysis': {}
        }
        
        # 간단히 평균+표준편차를 임계치로 사용
        threshold = np.mean(list(joint_differences.values())) + np.std(list(joint_differences.values()))
        
        for joint, diff in joint_differences.items():
            if math.isnan(diff):
                # NaN인 경우, 피드백 생성에서 제외하거나 별도의 처리
                feedback['detailed_analysis'][joint] = {
                    'difference': None,
                    'feedback': f"{joint.replace('_', ' ').title()}의 움직임에 데이터가 충분하지 않아 분석할 수 없습니다."
                }
                continue  # 혹은 다른 처리 방식 선택 가능
            
            if diff > threshold:
                feedback['major_differences'].append(joint)
            feedback['detailed_analysis'][joint] = {
                'difference': float(diff) if not math.isnan(diff) else None,  # NaN은 None으로 대체
                'feedback': self._generate_joint_feedback(joint, diff, threshold)
            }
        
        return feedback

    def _generate_joint_feedback(self, joint, difference, threshold):
        """개별 관절에 대한 구체적 피드백 생성"""
        if difference is None:
            return f"{joint.replace('_', ' ').title()}의 움직임을 분석할 수 없습니다."
        
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
    
    # --------------------------------------------------------------
    # --------------------------------------------------------------


    def analyze_shooting_mechanics(self, user_sequence, fps, output_video_path):
        """사용자의 슈팅 동작을 분석하고 프로 선수와 비교"""
        print("\nStarting shooting mechanics analysis...", file=sys.stderr)
        
        # MediaPipe pose detector 초기화
        pose_detector = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        features = []
        
        print("Extracting pose features...", file=sys.stderr)
        for frame_idx, frame in enumerate(user_sequence):
            try:
                # RGB 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 포즈 추출
                results = pose_detector.process(frame_rgb)
                
                if results.pose_landmarks:
                    # 랜드마크를 리스트로 변환
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                    features.append(landmarks)
                else:
                    print(f"No pose detected in frame {frame_idx}", file=sys.stderr)
            except Exception as e:
                print(f"Exception processing frame {frame_idx}: {e}", file=sys.stderr)
        
        if not features:
            print("No pose features detected", file=sys.stderr)
            return None
        
        print(f"Extracted features from {len(features)} frames", file=sys.stderr)
        features = np.array(features, dtype=np.float32)
        
        # 왼손잡이인 경우 데이터 변환
        if self.handedness == 'left':
            features = self.transform_to_right_handed(features)
        
        print("Running model inference...", file=sys.stderr)
        with torch.no_grad():
            user_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
            reconstructed = self.model(user_tensor)
            
            print("Generating overlay video...", file=sys.stderr)
            reconstructed_np = reconstructed.cpu().numpy()[0]

            normalized_reconstructed = normalize_reconstructed_landmarks(features, reconstructed_np)
            normalized_tensor = torch.FloatTensor(normalized_reconstructed).unsqueeze(0).to(device)

            differences = torch.abs(normalized_tensor - user_tensor)
            differences = differences.cpu().numpy()
            final_score = self.analyze_shooting_form(features, differences)
            
            if self.handedness == 'left':
                features = self.transform_to_right_handed(features)
                normalized_reconstructed = self.transform_to_right_handed(normalized_reconstructed)

            overlay_and_save_video(user_sequence, normalized_reconstructed, features, 
                                output_path=output_video_path, fps=fps)
            
        
        feedback = self.generate_feedback(differences[0])
        similarity = 1 - np.mean(differences)
        feedback['overall_similarity'] = float(similarity) * 100
        
        if self.handedness == 'left' and feedback is not None:
            feedback = self.swap_feedback_labels(feedback)
        
        print("Analysis completed", file=sys.stderr)
        # 추가로 output_video_path도 결과에 포함
        feedback['output_video_path'] = output_video_path
        
        return feedback

def detect_basketball_in_video(frames):
   """전체 프레임에서 농구공 검출 여부 확인"""
   ball_model = YOLO('best_2.pt')
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ball_model.to(device)

   for frame in frames:
       ball_results = ball_model.predict(frame, imgsz=1280, conf=0.5, verbose=False)
       ball_det = ball_results[0].boxes
       
       if torch.cuda.is_available():
           ball_boxes = ball_det.xyxy.cpu().numpy()
       else:
           ball_boxes = ball_det.xyxy.numpy()
           
       if len(ball_boxes) > 0:
           return True
   return False

def analyze_video(video_path, output_video_path, model_path='best_pretrain_model_train_1.pth', handedness='right'):
    """비디오 파일 분석"""
    try:
        video_path = os.path.abspath(video_path)
        output_video_path = os.path.abspath(output_video_path)

        if not os.path.exists(video_path):
            print(f"Error: Video file does not exist: {video_path}", file=sys.stderr)
            return None

        print(f"Opening video file: {video_path}", file=sys.stderr)
        analyzer = BasketballShootingAnalyzer(model_path, handedness=handedness)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}", file=sys.stderr)
            return None

        # 비디오 정보 출력
        fps = cap.get(cv2.CAP_PROP_FPS)  # 원본 FPS 사용
        frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_count += 1

        cap.release()
        if not detect_basketball_in_video(frames):
            print("No basketball detected in video", file=sys.stderr)
            return None
        
        if not frames:
            print("No frames were read from the video", file=sys.stderr)
            return None

        print(f"Successfully read {frame_count} frames from video", file=sys.stderr)
        feedback = analyzer.analyze_shooting_mechanics(frames, fps=fps, output_video_path=output_video_path)
        return feedback
    except Exception as e:
        print(f"Exception in analyze_video: {e}", file=sys.stderr)
        return None

def main():
    """
    shot_analyzer.py input_video_path output_video_path [handedness] 형태로 실행
    예) python shot_analyzer.py /path/to/inputs/input.mp4 /path/to/outputs/output.mp4 right
    """
    if len(sys.argv) < 3:
        print("Usage: python shot_analyzer.py <input_video_path> <output_video_path> [handedness (right/left)]", file=sys.stderr)
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_video_path = sys.argv[2]
    handedness = sys.argv[3] if len(sys.argv) > 3 else 'right'
    
    try:
        # 분석 수행
        feedback = analyze_video(video_path, output_video_path, handedness=handedness)
        
        # 결과 출력 (JSON)
        if feedback is None:
            # 포즈 분석 실패(혹은 에러) -> 에러 표준출력 대신, JSON으로 알림
            result = {
                "success": False,
                "message": "No pose detected or video read failed."
            }
            print(json.dumps(result))  # Uncommented
            sys.exit(0)
        else:
            # 정상 분석
            result = {
                "success": True,
                "overall_similarity": feedback["overall_similarity"],
                "major_differences": feedback["major_differences"],
                "detailed_analysis": feedback["detailed_analysis"],
                "output_video_path": output_video_path
            }
            print(json.dumps(result))
    except Exception as e:
        # 모든 예외를 캐치하고 JSON 형태로 출력
        result = {
            "success": False,
            "message": str(e)
        }
        print(json.dumps(result))  # Uncommented
        sys.exit(1)

if __name__ == "__main__":
    main()
