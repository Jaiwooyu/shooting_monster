import os
import time
import json
import logging
import cv2
import numpy as np
import mediapipe as mp
import yt_dlp
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import torch
from ultralytics import YOLO
from norfair import Detection, Tracker
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class BasketballDetector:
    def __init__(self):
        # Constants
        self.MIN_CONF = 0.4
        self.MIN_IOU = 0.45
        self.DIST_THRESHOLD_PLAYER = 50
        self.DIST_THRESHOLD_BALL = 50
        self.BALL_HAND_THRESHOLD = 70
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Shooting detection parameters
        self.shooting_thresholds = {
            'min_sequence_length': 15,  # Minimum frames for a shooting sequence
            'elbow_angle_threshold': 150,  # Minimum elbow angle for shooting
            'wrist_height_threshold': 0.8,  # Relative to shoulder height
            'ball_distance_threshold': 50  # Maximum distance between ball and hands
        }
        
        # Initialize trackers
        self.player_tracker = Tracker(
            distance_function=self.euclidean_distance,
            distance_threshold=self.DIST_THRESHOLD_PLAYER,
            initialization_delay=0,
            hit_counter_max=30,
        )
        
        self.ball_tracker = Tracker(
            distance_function=self.euclidean_distance,
            distance_threshold=self.DIST_THRESHOLD_BALL,
            initialization_delay=0,
            hit_counter_max=30,
        )
        
        # Load models
        self.player_model = YOLO("yolov8l.pt")
        self.ball_model = YOLO("basketballModel.pt")
        
    def setup_mediapipe(self):
        """MediaPipe 초기화"""
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            filename=self.output_dir / 'collection.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def setup_directories(self):
        """디렉토리 생성"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_thresholds(self):
        """슈팅 동작 감지 임계값 설정"""
        self.shooting_thresholds = {
            'elbow_angle': (150, 180),
            'knee_angle': (130, 180),
            'wrist_height': 0.7,
            'min_sequence_length': 15
        }
    
    def search_videos(self, query: str) -> List[str]:
        """YouTube 검색 실행"""
        try:
            search_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
                'format': 'best'  # 검색 시에는 format 제한 없이
            }
            
            search_url = f"ytsearch{self.max_videos_per_query}:{query}"
            with yt_dlp.YoutubeDL(search_opts) as ydl:
                result = ydl.extract_info(search_url, download=False)
                if 'entries' in result:
                    # URL 리스트 생성 (이미 처리된 비디오 제외)
                    video_urls = [
                        f"https://youtube.com/watch?v={entry['id']}"
                        for entry in result['entries']
                        if entry is not None and
                        f"https://youtube.com/watch?v={entry['id']}" not in self.processed_videos
                    ]
                    return video_urls
            return []
        except Exception as e:
            logging.error(f"Error searching for query '{query}': {e}")
            return []

    def download_video(self, url: str) -> Optional[str]:
        """비디오 다운로드"""
        try:
            # 먼저 사용 가능한 포맷 확인
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                formats = info.get('formats', [])
                
                # 720p 이상의 포맷 찾기
                suitable_formats = [
                    f for f in formats 
                    if f.get('height', 0) and f['height'] >= 720 or 
                    (isinstance(f.get('format_note', ''), str) and 
                        'p' in f.get('format_note', '') and 
                        int(f.get('format_note', '').replace('p', '')) >= 720)
                ]
                
                # 적합한 포맷이 없으면 가장 높은 해상도 선택
                if not suitable_formats and formats:
                    suitable_formats = sorted(
                        formats,
                        key=lambda x: (x.get('height', 0) or 0, x.get('filesize', 0) or 0),
                        reverse=True
                    )
                
                if not suitable_formats:
                    logging.error(f"No suitable format found for {url}")
                    return None
                
                # 가장 적합한 포맷 선택
                best_format = suitable_formats[0]
                format_id = best_format['format_id']
            
            # 선택된 포맷으로 다운로드
            download_opts = {
                'format': format_id,
                'outtmpl': str(self.temp_dir / '%(id)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(download_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = self.temp_dir / f"{info['id']}.{info['ext']}"
                return str(video_path)
                
        except Exception as e:
            logging.error(f"Error downloading video {url}: {e}")
            return None

        
    def calculate_angle(self, a: Tuple[float, float], 
                       b: Tuple[float, float], 
                       c: Tuple[float, float]) -> float:
        """세 점 사이의 각도 계산"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                 np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def is_shooting_motion(self, landmarks, ball_center=None):
        """Detect if the pose represents a shooting motion"""
        if not landmarks:
            return False
            
        # Get relevant landmarks
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        # Calculate arm angle
        shoulder_pos = (r_shoulder.x, r_shoulder.y)
        elbow_pos = (r_elbow.x, r_elbow.y)
        wrist_pos = (r_wrist.x, r_wrist.y)
        
        arm_angle = self.calculate_angle(shoulder_pos, elbow_pos, wrist_pos)
        
        # Check if wrist is above shoulder
        wrist_above_shoulder = r_wrist.y < r_shoulder.y
        
        # Check if ball is near hands (if ball position is provided)
        ball_near_hands = True
        if ball_center is not None:
            wrist_pos_px = (int(r_wrist.x), int(r_wrist.y))
            ball_distance = np.linalg.norm(np.array(wrist_pos_px) - np.array(ball_center))
            ball_near_hands = ball_distance < self.shooting_thresholds['ball_distance_threshold']
        
        # Combined conditions for shooting motion
        return (arm_angle > self.shooting_thresholds['elbow_angle_threshold'] and 
                wrist_above_shoulder and 
                ball_near_hands)

    # 슈팅 동작의 영상 프레임을 저장하는 메서드
    def save_sequence(self, frame_buffer: List[Tuple], sequence_id: int):
        """슈팅 시퀀스 저장"""
        sequence_dir = self.output_dir / f"shot_{sequence_id}"
        sequence_dir.mkdir(exist_ok=True)
        
        # 프레임 및 랜드마크 저장
        landmarks_data = []
        landmarks_list = []  # 품질 점수 계산용
        
        for i, (frame, landmarks) in enumerate(frame_buffer):
            # 프레임 저장
            cv2.imwrite(str(sequence_dir / f"frame_{i:03d}.jpg"), frame)
            
            # 랜드마크 데이터 수집
            frame_landmarks = []
            for lm in landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.visibility])
            landmarks_data.append(frame_landmarks)
            landmarks_list.append(landmarks.landmark)
            
            
    def collect_shots(self, num_threads: int = 4):
        """메인 수집 프로세스"""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=self.max_training_hours)
        
        while datetime.now() < end_time:
            try:
                # 각 검색어에 대해 비디오 수집
                for query in self.search_queries:
                    video_urls = self.search_videos(query)
                    
                    if not video_urls:
                        logging.info(f"No new videos found for query: {query}")
                        continue
                    
                    # 비디오 처리
                    with ThreadPoolExecutor(max_workers=num_threads) as executor:
                        futures = []
                        for url in video_urls:
                            futures.append(
                                executor.submit(self.process_single_video, url)
                            )
                        
                        # 결과 수집
                        for future in futures:
                            try:
                                future.result()  # 에러 체크
                            except Exception as e:
                                logging.error(f"Error in video processing: {e}")
                    
                    # 진행 상황 출력
                    self.print_progress(end_time)
                
                # 잠시 대기
                time.sleep(10)  # 검색 간 대기 시간
                
            except Exception as e:
                logging.error(f"Error in collection process: {e}")
                continue
        self.cleanup()
    
    def process_single_video(self, url: str):
        """단일 비디오 처리"""
        try:
            video_path = self.download_video(url)
            if video_path:
                try:
                    self.process_video(video_path)
                    self.processed_videos.add(url)
                finally:
                    if os.path.exists(video_path):
                        os.remove(video_path)
        except Exception as e:
            logging.error(f"Error processing video {url}: {e}")


    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_buffer = []
        sequence_count = 0
        
        progress_bar = tqdm(total=total_frames, desc="Processing Video")
        
        with self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose_detector:
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect players and balls
                player_results = self.player_model.predict(
                    source=frame,
                    conf=self.MIN_CONF,
                    iou=self.MIN_IOU,
                    device=0,
                    verbose=False
                )[0]

                ball_results = self.ball_model.predict(
                    source=frame,
                    conf=self.MIN_CONF,
                    iou=self.MIN_IOU,
                    device=0,
                    verbose=False
                )[0]

                # Process detections
                person_detections = []
                ball_detections = []

                # Process player detections
                if player_results.boxes is not None:
                    for box in player_results.boxes:
                        if int(box.cls[0].item()) == 0:  # person class
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            cx, cy = ((x1 + x2)/2, (y1 + y2)/2)
                            person_detections.append(
                                Detection(points=np.array([[cx, cy]]),
                                        data={'bbox':[x1, y1, x2, y2]})
                            )

                # Process ball detections
                if ball_results.boxes is not None:
                    for box in ball_results.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx, cy = ((x1 + x2)/2, (y1 + y2)/2)
                        ball_detections.append(
                            Detection(points=np.array([[cx, cy]]),
                                    data={'bbox':[x1, y1, x2, y2]})
                        )

                # Update trackers
                tracked_players = self.player_tracker.update(person_detections)
                tracked_balls = self.ball_tracker.update(ball_detections)

                # Get ball center
                ball_center = None
                for ball in tracked_balls:
                    if ball.last_detection is not None:
                        x1b, y1b, x2b, y2b = ball.last_detection.data['bbox']
                        ball_center = ((x1b + x2b)/2, (y1b + y2b)/2)
                        break

                # Process nearest player
                if ball_center is not None:
                    nearest_player = None
                    min_dist = float('inf')

                    for player in tracked_players:
                        if player.last_detection is not None:
                            x1p, y1p, x2p, y2p = player.last_detection.data['bbox']
                            player_center = ((x1p + x2p)/2, (y1p + y2p)/2)
                            dist = np.hypot(player_center[0] - ball_center[0],
                                          player_center[1] - ball_center[1])
                            if dist < min_dist:
                                min_dist = dist
                                nearest_player = player

                    # Process pose for nearest player
                    if nearest_player and nearest_player.last_detection is not None:
                        x1p, y1p, x2p, y2p = nearest_player.last_detection.data['bbox']
                        person_roi = frame_rgb[int(y1p):int(y2p), int(x1p):int(x2p)]
                        
                        if person_roi.size > 0:
                            results_pose = pose_detector.process(person_roi)
                            
                            if results_pose.pose_landmarks:
                                # Scale landmarks to original frame coordinates
                                h, w = person_roi.shape[:2]
                                scaled_landmarks = []
                                for landmark in results_pose.pose_landmarks.landmark:
                                    scaled_landmarks.append(type('LandmarkWrapper', (), {
                                        'x': landmark.x * w + x1p,
                                        'y': landmark.y * h + y1p,
                                        'z': landmark.z,
                                        'visibility': landmark.visibility
                                    }))
                                
                                # Check for shooting motion
                                if self.is_shooting_motion(scaled_landmarks, ball_center):
                                    frame_buffer.append((frame, results_pose.pose_landmarks))
                                else:
                                    # Save sequence if buffer is full
                                    if len(frame_buffer) >= self.shooting_thresholds['min_sequence_length']:
                                        self.save_sequence(frame_buffer, sequence_count)
                                        sequence_count += 1
                                    frame_buffer = []

                progress_bar.update(1)

        progress_bar.close()
        cap.release()
        print(f"[INFO] Processing complete. Found {sequence_count} shooting sequences.")

        
    def print_progress(self, end_time: datetime):
        """진행 상황 출력"""
        now = datetime.now()
        remaining_time = end_time - now
        
        print("\n=== Collection Progress ===")
        print(f"Processed Videos: {len(self.processed_videos)}")
        print(f"Remaining Time: {remaining_time}")
        print(f"Total Sequences: {len(list(self.output_dir.glob('shot_*')))}")
        print("=========================\n")
        
    def cleanup(self):
        """임시 파일 정리"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


def main():
    search_queries = [
        "KBL 하이라이트",  # 명확한 슈팅 폼 영상 위주로
    ]
    
    # 수집기 초기화 및 실행
    collector = BasketballShotDataCollector(
        output_dir="basketball_shot_dataset",
        max_training_hours=1,           # 1시간으로 제한
        search_queries=search_queries,
        max_videos_per_query=10,         # 쿼리당 3개만 처리
        min_resolution=720,
        sequence_length=30
    )
    
    collector.collect_shots(num_threads=2)  # 스레드 수도 줄임

if __name__ == "__main__":
    main()