import os
import time
import json
import logging
import cv2
import numpy as np
import mediapipe as mp
import yt_dlp
import torch
from ultralytics import YOLO
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class ShootingDetector:
    def __init__(self, frame_rate=30):
        self.ball_history = []  # 공의 위치 이력 저장
        self.base_frame_rate = 30  # 기준 프레임 레이트
        self.current_frame_rate = frame_rate
        self.frame_rate_ratio = frame_rate / self.base_frame_rate
        
        # 프레임 레이트에 따라 조정되는 임계값
        self.shooting_threshold = {
            'vertical_velocity': 15 * self.frame_rate_ratio,     # 수직 속도 임계값
            'horizontal_velocity': 20 * self.frame_rate_ratio,   # 수평 속도 임계값
            'trajectory_points': max(3, int(5 * self.frame_rate_ratio)),  # 궤적 분석을 위한 최소 포인트 수
            'min_height_change': 50,     # 최소 수직 이동 거리
            'min_horizontal_change': 30,  # 최소 수평 이동 거리
            'max_horizontal_change': 400, # 최대 수평 이동 거리
            'detection_window': max(5, int(10 * self.frame_rate_ratio))  # 분석할 프레임 수
        }

    def analyze_ball_trajectory(self, ball_box):
        """공의 궤적을 분석하여 슈팅 동작 감지"""
        if ball_box is None:
            self.ball_history.append(None)
            return False

        # 공의 중심점 계산
        x1, y1, x2, y2 = ball_box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 위치 이력 저장
        self.ball_history.append((center_x, center_y))
        
        # 이력이 충분하지 않으면 False 반환
        if len(self.ball_history) < self.shooting_threshold['trajectory_points']:
            return False

        # 최근 N개의 포인트만 분석
        recent_points = [p for p in self.ball_history[-self.shooting_threshold['detection_window']:] if p is not None]
        if len(recent_points) < self.shooting_threshold['trajectory_points']:
            return False

        # 수직 및 수평 움직임 분석
        vertical_movements = [recent_points[i][1] - recent_points[i-1][1] 
                            for i in range(1, len(recent_points))]
        horizontal_movements = [recent_points[i][0] - recent_points[i-1][0] 
                             for i in range(1, len(recent_points))]
        
        # 전체 이동 거리 계산
        total_vertical_change = abs(recent_points[-1][1] - recent_points[0][1])
        total_horizontal_change = abs(recent_points[-1][0] - recent_points[0][0])
        
        # 평균 속도 계산
        avg_vertical_velocity = abs(sum(vertical_movements) / len(vertical_movements))
        avg_horizontal_velocity = abs(sum(horizontal_movements) / len(horizontal_movements))

        # 슈팅 판정 조건:
        # 1. 수직 속도가 임계값을 넘는가
        # 2. 수평 속도가 적절한 범위 내인가
        # 3. 전체 수직 이동 거리가 충분한가
        # 4. 수평 이동 거리가 적절한 범위 내인가
        # 5. 대체로 위로 향하는 움직임인가
        is_shooting = (
            avg_vertical_velocity > self.shooting_threshold['vertical_velocity'] and
            avg_horizontal_velocity < self.shooting_threshold['horizontal_velocity'] and
            total_vertical_change > self.shooting_threshold['min_height_change'] and
            self.shooting_threshold['min_horizontal_change'] <= total_horizontal_change <= self.shooting_threshold['max_horizontal_change'] and
            sum(1 for v in vertical_movements if v < 0) > len(vertical_movements) * 0.6  # 60% 이상이 위로 향하는 움직임
        )

        # 이력 관리 (최근 N개만 유지)
        if len(self.ball_history) > self.shooting_threshold['detection_window']:
            self.ball_history.pop(0)

        return is_shooting


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
        
        self.search_queries = search_queries
        
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
    
    def calculate_trunk_angle(shoulder_mid, hip_mid):
        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]
        angle_deg = np.degrees(np.arctan2(dy, dx))  
        trunk_angle = angle_deg - 90
        return trunk_angle
    
    def save_sequence(self, sequence_buffer, sequence_count, sequence_metadata=None):
        """시퀀스 저장"""
        sequence_dir = os.path.join(self.output_dir, f"sequence_{sequence_count}")
        os.makedirs(sequence_dir, exist_ok=True)
        
        # 포즈와 박스 정보를 저장할 메타데이터 리스트
        frame_metadata = []
        
        for i, frame_info in enumerate(sequence_buffer):
            # 프레임 저장
            frame_path = os.path.join(sequence_dir, f"frame_{i:03d}.jpg")
            cv2.imwrite(frame_path, frame_info['frame'])
            
            # 메타데이터 수집
            frame_metadata.append({
                'frame_index': i,
                'ball_box': frame_info['ball_box'],
                'frame_idx': frame_info.get('frame_idx', i)  # 원본 프레임 인덱스 저장
            })
        
        # 전체 메타데이터 구성
        metadata = {
            'frames': frame_metadata,
            'sequence_info': sequence_metadata or {}  # 시퀀스 레벨 메타데이터 추가
        }
        
        # 메타데이터를 JSON 파일로 저장
        metadata_path = os.path.join(sequence_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

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

        # 비디오 속성 가져오기
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 프레임 레이트에 따른 버퍼 크기 조정
        fps_ratio = fps / 30.0
        PRE_FRAMES = max(10, int(20 * fps_ratio))  # 슈팅 감지 전 저장할 프레임 수
        POST_FRAMES = max(5, int(10 * fps_ratio))  # 슈팅 감지 후 저장할 프레임 수

        # 임시 버퍼와 실제 시퀀스 버퍼 초기화
        temp_buffer = []  # 최근 프레임을 저장할 임시 버퍼
        sequence_buffer = []  # 실제 슈팅 시퀀스를 저장할 버퍼
        sequence_count = 0
        
        is_capturing = False  # 슈팅 시퀀스 캡처 중인지 확인하는 플래그
        post_frame_count = 0  # 슈팅 감지 후 저장된 프레임 수

        # 모델 초기화
        ball_model = YOLO('best_2.pt')
        ball_model.to(device)
        shooting_detector = ShootingDetector(frame_rate=fps)  # fps 정보 전달

        # MediaPipe Pose 초기화
        pose_estimator = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 프로그레스 바 설정
        progress_bar = tqdm(total=total_frames, desc=f"Processing {Path(video_path).name}")
        frame_idx = 0
        
        # 메타데이터 저장을 위한 정보
        sequence_metadata = {
            'fps': fps,
            'width': width,
            'height': height,
            'video_path': video_path
        }

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 이미지 전처리
            if torch.cuda.is_available():
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_chw = frame_rgb.transpose(2, 0, 1)
                frame_cuda = torch.from_numpy(frame_chw).unsqueeze(0).to(device)
            else:
                frame_cuda = frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 농구공 탐지
            ball_results = ball_model.predict(
                frame, imgsz=1280, conf=0.1, verbose=False
            )
            ball_det = ball_results[0].boxes

            # 박스 좌표 처리
            ball_boxes = ball_det.xyxy.cpu().numpy() if torch.cuda.is_available() else ball_det.xyxy.numpy()
            ball_box = None
            max_ball_area = 0

            # 가장 큰 농구공 박스 선택
            for j in range(len(ball_boxes)):
                x1, y1, x2, y2 = ball_boxes[j]
                area = (x2 - x1) * (y2 - y1)
                if area > max_ball_area:
                    max_ball_area = area
                    ball_box = (int(x1), int(y1), int(x2), int(y2))

            # 현재 프레임 정보를 임시 버퍼에 저장
            current_frame_info = {
                'frame': frame,
                'ball_box': ball_box,
                'frame_idx': frame_idx
            }

            temp_buffer.append(current_frame_info)
            
            # 임시 버퍼 크기 유지
            if len(temp_buffer) > PRE_FRAMES:
                temp_buffer.pop(0)

            # 슈팅 동작 감지
            if ball_box is not None:
                is_shooting = shooting_detector.analyze_ball_trajectory(ball_box)
                
                if is_shooting and not is_capturing:
                    # 슈팅이 감지되면 임시 버퍼의 내용을 시퀀스 버퍼로 복사
                    sequence_buffer.extend(temp_buffer.copy())
                    is_capturing = True
                    post_frame_count = 0
                
                if is_capturing:
                    sequence_buffer.append(current_frame_info)
                    post_frame_count += 1
                    
                    # 슈팅 후 지정된 프레임 수만큼 캡처했으면 시퀀스 저장
                    if post_frame_count >= POST_FRAMES:
                        if len(sequence_buffer) >= self.shooting_thresholds['min_sequence_length']:
                            self.save_sequence(sequence_buffer, sequence_count, sequence_metadata)
                            sequence_count += 1
                        sequence_buffer = []
                        is_capturing = False
                        post_frame_count = 0

            frame_idx += 1
            progress_bar.update(1)

        # 마지막 시퀀스 처리
        if len(sequence_buffer) > 0:
            self.save_sequence(sequence_buffer, sequence_count, sequence_metadata)
            sequence_count += 1

        progress_bar.close()
        cap.release()
        logging.info(f"Processed {frame_idx} frames, found {sequence_count} shooting sequences")
        return sequence_count
        

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
        "kbl 김낙현 하이라이트",
        "kbl 벨란겔 하이라이트",
        "kbl 유기상 하이라이트",
        "kbl 이근휘 하이라이트",
        "kbl 이우석 하이라이트",
        "kbl 이재도 하이라이트",
        "kbl 이정현 하이라이트",
        "kbl 배병준 하이라이트",
        "kbl 허웅 하이라이트",
        "kbl 허훈 하이라이트",
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