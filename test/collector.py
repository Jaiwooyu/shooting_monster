import os
import time
from datetime import datetime, timedelta
from pytube import YouTube, Search
import cv2
from tempfile import NamedTemporaryFile
import shutil
import numpy as np
from .utils import (calculate_angle, check_elbow_quality, check_knee_quality,
                   check_balance_quality, check_shot_form_quality, analyze_shooting_sequence,
                   detect_ball)

class BasketballShotCollector:
    def __init__(self, max_training_hours=12):
        self.temp_dir = "temp_videos"
        self.output_dir = "processed_shots"
        self.max_training_hours = max_training_hours
        self.start_time = None
        
        # OpenPose 초기화
        self.datum, self.opWrapper = self.initialize_openpose()
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 품질 기준 설정
        self.quality_threshold = 0.8  # 80% 이상의 품질 점수
        self.min_sequence_length = 15  # 최소 0.5초(30fps 기준)

    def initialize_openpose(self):
        try:
            if platform == "win32":
                sys.path.append(os.path.dirname(os.getcwd()))
                import OpenPose.Release.pyopenpose as op
            else:
                sys.path.append('/usr/local/python')
                import pyopenpose as op
                
            params = dict()
            params["model_folder"] = "./OpenPose/models"
            opWrapper = op.WrapperPython()
            opWrapper.configure(params)
            opWrapper.start()
            return op.Datum(), opWrapper
            
        except Exception as e:
            print(f"Error initializing OpenPose: {e}")
            raise

    def is_shooting_motion(self, frame):
        self.datum.cvInputData = frame
        self.opWrapper.emplaceAndPop([self.datum])
        
        try:
            keypoints = self.datum.poseKeypoints[0]
            
            # 동작 품질 평가
            form_quality = check_shot_form_quality(keypoints)
            
            # 공 감지
            ball_detected = detect_ball(frame)
            
            # 슛 동작 판단
            is_shooting = (
                form_quality > self.quality_threshold and
                ball_detected and
                self.check_shooting_pose(keypoints)
            )
            
            return is_shooting
        except:
            return False

    def check_shooting_pose(self, keypoints):
        # 슛 동작의 기본 자세 확인
        elbow_score = check_elbow_quality(keypoints)
        knee_score = check_knee_quality(keypoints)
        balance_score = check_balance_quality(keypoints)
        
        # 모든 점수가 기준 이상인지 확인
        return all([
            elbow_score > 0.6,
            knee_score > 0.6,
            balance_score > 0.6
        ])

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        shot_sequence = []
        keypoints_sequence = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            self.datum.cvInputData = frame
            self.opWrapper.emplaceAndPop([self.datum])
            
            try:
                keypoints = self.datum.poseKeypoints[0]
                if self.is_shooting_motion(frame):
                    shot_sequence.append(frame)
                    keypoints_sequence.append(keypoints)
                else:
                    if len(shot_sequence) > self.min_sequence_length:
                        # 시퀀스 전체 품질 분석
                        sequence_analysis = analyze_shooting_sequence(keypoints_sequence)
                        if sequence_analysis['average_score'] > self.quality_threshold:
                            self.save_sequence(shot_sequence, sequence_analysis)
                    shot_sequence = []
                    keypoints_sequence = []
            except:
                continue
                
        cap.release()

    def save_sequence(self, sequence, analysis):
        sequence_id = len(os.listdir(self.output_dir))
        sequence_dir = os.path.join(self.output_dir, f"shot_{sequence_id}")
        os.makedirs(sequence_dir, exist_ok=True)
        
        # 프레임 저장
        for i, frame in enumerate(sequence):
            cv2.imwrite(os.path.join(sequence_dir, f"frame_{i:03d}.jpg"), frame)
        
        # 분석 결과 저장
        analysis_path = os.path.join(sequence_dir, 'analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump({
                'average_score': float(analysis['average_score']),
                'max_score': float(analysis['max_score']),
                'min_score': float(analysis['min_score']),
                'consistency': float(analysis['consistency']),
                'timestamp': datetime.now().isoformat()
            }, f, indent=4)

    def print_progress(self, search_queries, end_time):
        elapsed_time = datetime.now() - self.start_time
        remaining_time = end_time - datetime.now()
        
        print("\n=== Collection Progress ===")
        print(f"Elapsed Time: {elapsed_time}")
        print(f"Remaining Time: {remaining_time}")
        print("\nVideos processed per query:")
        for query, count in search_queries.items():
            print(f"{query}: {count}")
        print(f"\nTotal sequences collected: {len(os.listdir(self.output_dir))}")
        print("=========================\n")

    def cleanup(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)