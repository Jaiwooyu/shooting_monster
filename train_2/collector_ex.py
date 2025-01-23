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

class BasketballShotDataCollector:
    def __init__(self,
                 output_dir: str = "shot_dataset",
                 max_training_hours: int = 12,
                 search_queries: List[str] = None,
                 max_videos_per_query: int = 50,
                 min_resolution: int = 720,
                 sequence_length: int = 30):
        
        self.output_dir = Path(output_dir)
        self.temp_dir = self.output_dir / "temp_videos"
        self.max_training_hours = max_training_hours
        self.sequence_length = sequence_length
        
        self.search_queries = search_queries or [
            "NBA basketball highlights",
            "NBA shooting form",
            "basketball shooting training",
            "basketball game highlights",
            "basketball shooting technique"
        ]
        
        self.max_videos_per_query = max_videos_per_query
        self.min_resolution = min_resolution
        self.processed_videos = set()
        
        # yt-dlp 기본 옵션 설정
        self.ydl_opts = {
            'format': 'bestvideo[height>=720][ext=mp4]+bestaudio[ext=m4a]/best[height>=720]/best',
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,  # 검색 결과용
        }
        
        # MediaPipe 초기화 및 기타 설정은 동일하게 유지
        self.setup_mediapipe()
        self.setup_logging()
        self.setup_directories()
        self.setup_thresholds()
        
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
    
    def is_shooting_motion(self, landmarks) -> bool:
        """슈팅 동작 감지"""
        try:
            # 필요한 랜드마크 추출
            r_shoulder = (landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
            r_elbow = (landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y)
            r_wrist = (landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y)
            r_hip = (landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y)
            r_knee = (landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                     landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y)
            r_ankle = (landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)
            
            # 각도 계산
            elbow_angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
            knee_angle = self.calculate_angle(r_hip, r_knee, r_ankle)
            
            # 손목 높이 확인 (어깨 대비)
            wrist_above_shoulder = r_wrist[1] < r_shoulder[1] * self.shooting_thresholds['wrist_height']
            
            # 슈팅 동작 판정
            is_shooting = (
                self.shooting_thresholds['elbow_angle'][0] <= elbow_angle <= self.shooting_thresholds['elbow_angle'][1] and
                self.shooting_thresholds['knee_angle'][0] <= knee_angle <= self.shooting_thresholds['knee_angle'][1] and
                wrist_above_shoulder
            )
            
            return is_shooting
            
        except Exception as e:
            logging.error(f"Error in shooting motion detection: {e}")
            return False
        
    def calculate_shot_quality(self, landmarks_list: List) -> float:
        """슈팅 품질 점수 계산"""
        quality_scores = []
        
        for landmarks in landmarks_list:
            try:
                # 관절 각도 계산
                r_shoulder = (landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
                r_elbow = (landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y)
                r_wrist = (landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y)
                r_hip = (landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y)
                r_knee = (landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y)
                r_ankle = (landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)

                # 각도 계산
                elbow_angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
                knee_angle = self.calculate_angle(r_hip, r_knee, r_ankle)

                # 각도 기반 점수 계산 (0-1 범위)
                elbow_score = 1.0 - abs(elbow_angle - 165) / 90  # 이상적인 팔꿈치 각도는 약 165도
                knee_score = 1.0 - abs(knee_angle - 150) / 90    # 이상적인 무릎 각도는 약 150도
                
                # 손목 높이 점수
                wrist_height_score = 1.0 if r_wrist[1] < r_shoulder[1] else 0.5

                # 전체 점수 계산
                frame_score = (elbow_score * 0.4 + knee_score * 0.3 + wrist_height_score * 0.3)
                quality_scores.append(max(0.0, min(1.0, frame_score)))  # 0-1 범위로 제한

            except Exception as e:
                logging.error(f"Error calculating shot quality: {e}")
                quality_scores.append(0.0)

        # 전체 시퀀스의 평균 점수 계산
        return np.mean(quality_scores)
        
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
        
        # 슈팅 품질 점수 계산
        quality_score = self.calculate_shot_quality(landmarks_list)
        
        # analysis.json 저장 (프로세서 형식에 맞춤)
        analysis = {
            'average_score': float(quality_score),
            'max_score': float(np.max(quality_score) if isinstance(quality_score, np.ndarray) else quality_score),
            'min_score': float(np.min(quality_score) if isinstance(quality_score, np.ndarray) else quality_score),
            'consistency': float(np.std(quality_score) if isinstance(quality_score, np.ndarray) else 0.0),
            'num_frames': len(frame_buffer),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(sequence_dir / 'analysis.json', 'w') as f:
            json.dump(analysis, f, indent=4)
            
        # 추가 메타데이터 저장 (선택사항)
        metadata = {
            'landmarks': landmarks_data,
            'sequence_id': sequence_id
        }
        
        with open(sequence_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
            
            
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

    def process_video(self, video_path: str):
        """비디오에서 슈팅 시퀀스 추출"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Error opening video: {video_path}")
            return
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_buffer = []
        sequence_count = 0
        
        # Pose detector를 여기서 새로 초기화
        with self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose_detector:
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # RGB 변환 및 포즈 추출
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 현재 프레임 타임스탬프 설정
                frame_rgb.flags.writeable = False
                results = pose_detector.process(frame_rgb)
                frame_rgb.flags.writeable = True
                
                if results.pose_landmarks:
                    # 슈팅 동작 감지
                    if self.is_shooting_motion(results.pose_landmarks.landmark):
                        frame_buffer.append((frame, results.pose_landmarks))
                    else:
                        # 슈팅 시퀀스 저장
                        if len(frame_buffer) >= self.shooting_thresholds['min_sequence_length']:
                            self.save_sequence(frame_buffer, sequence_count)
                            sequence_count += 1
                        frame_buffer = []
                
                frame_idx += 1
                    
        cap.release()
        
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
        "basketball shooting drill",  # 명확한 슈팅 폼 영상 위주로
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