import cv2
import csv
import math
import numpy as np
import mediapipe as mp
import torch
import warnings
from ultralytics import YOLO
from norfair import Detection, Tracker, draw_tracked_objects
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================
# 1) 설정 상수/파라미터
# ============================
MIN_CONF = 0.4       # YOLO confidence threshold
MIN_IOU  = 0.45      # YOLO IoU threshold
DIST_THRESHOLD_PLAYER = 50  # Norfair player distance threshold
DIST_THRESHOLD_BALL   = 50  # Norfair ball distance threshold

# "공 보유" 판정 거리(손목↔공)
BALL_HAND_THRESHOLD = 70
# "드리블" 로직을 위해, 공이 플레이어의 발끝 근처까지 떨어졌다가 일정 프레임 내 다시 올라오면 "드리블" 상태라고 가정
DRIBBLE_FOOT_THRESHOLD = 50
DRIBBLE_TIME_WINDOW    = 30  # 몇 프레임 내에 '공이 발 근처' → '손목 근처'가 일어나면 드리블

# ============================
# 2) MediaPipe Pose 준비
# ============================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ============================
# 3) 각도 계산, 슛폼 점수 등 함수
# ============================
def calculate_angle(a, b, c):
    """
    세 점(a, b, c)의 (x, y) 좌표를 받아 b를 꼭짓점으로 하는 0~180도 각도를 계산
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def get_bbox_center(bbox):
    """
    bbox = [x1, y1, x2, y2]
    중심점 (cx, cy) 반환
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2)/2, (y1 + y2)/2)

def score_shot_form(landmarks_px):
    """
    단순한 슛폼 스코어 예시 (오른팔/오른다리 기준).
    """
    # MediaPipe Right 관절
    r_shoulder = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    r_elbow    = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
    r_wrist    = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_WRIST.value)
    r_hip      = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_HIP.value)
    r_knee     = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_KNEE.value)
    r_ankle    = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

    if not all([r_shoulder, r_elbow, r_wrist, r_hip, r_knee, r_ankle]):
        return 0.0

    angle_arm = calculate_angle(r_shoulder, r_elbow, r_wrist)
    angle_leg = calculate_angle(r_hip, r_knee, r_ankle)

    ref_arm = 100.0
    ref_leg = 170.0
    err_arm = abs(angle_arm - ref_arm)
    err_leg = abs(angle_leg - ref_leg)
    total_err = err_arm + err_leg

    max_tol = 50.0
    score = 100.0 * max(0.0, (max_tol - total_err)) / max_tol
    return round(score, 1)

# ============================
# 4) Norfair Tracker 설정
# ============================
def euclidean_distance(detection, tracked_object):
    det_xy = detection.points[0]
    trk_xy = tracked_object.estimate[0]
    return np.linalg.norm(det_xy - trk_xy)

player_tracker = Tracker(
    distance_function=euclidean_distance,
    distance_threshold=DIST_THRESHOLD_PLAYER,
    initialization_delay=0,
    hit_counter_max=30,
)

ball_tracker = Tracker(
    distance_function=euclidean_distance,
    distance_threshold=DIST_THRESHOLD_BALL,
    initialization_delay=0,
    hit_counter_max=30,
)

# ============================
# 5) 메인 파이프라인
# ============================
def main():
    # --------------------------------
    # (A) YOLOv5 모델 로드
    # --------------------------------
    model = YOLO("basketballModel.pt")
    print("클래스 이름들:", model.names)
    
    # --------------------------------
    # (B) 입력 비디오 & 결과 비디오 설정
    # --------------------------------
    input_video = "dataset1/multi2.mov"
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error opening video: {input_video}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = "dataset1/output_mul2.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    print(f"[INFO] Video output: {out_path}")

    # --------------------------------
    # (C) CSV 로깅 설정
    # --------------------------------
    csv_filename = "dataset1/basketball_game_data.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "player_id", "ball_state",
            "angle_arm", "angle_leg", "shot_score"
        ])

    # --------------------------------
    # (D) MediaPipe Pose 준비
    # --------------------------------
    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frame_idx = 0
    ball_movement_history = {}
    progress_bar = tqdm(total=total_frames, desc="Processing Video")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] No more frames. Exiting...")
            break

        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # -------------------------------
        # 1) YOLOv5 추론
        # -------------------------------
        results = model.predict(
            source=frame,
            conf=MIN_CONF,
            iou=MIN_IOU,
            device=0,
            verbose=False
        )[0]

        ball_detections = []

        # YOLOv5 결과 처리
        if results.boxes is not None:
            for box in results.boxes:  # person_results.boxes -> ball_results.boxes로 수정
                # ball 클래스 번호는 학습된 모델에 따라 다를 수 있으므로 
                # best.pt 모델에서의 ball 클래스 인덱스를 사용해야 합니다
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = get_bbox_center([x1, y1, x2, y2])
                ball_detections.append(  # person_detections -> ball_detections로 수정
                    Detection(points=np.array([[cx, cy]]),
                            data={'bbox':[x1, y1, x2, y2]})
                )

        # [이하 코드는 동일하게 유지...]
        # Norfair tracking, pose estimation, ball state detection 등
        tracked_ball = player_tracker.update(ball_detections)

        # Visualization code...
        for trk_obj in tracked_ball:
            pid = trk_obj.global_id
            cx, cy = trk_obj.estimate[0]
            if trk_obj.last_detection is not None:
                x1,y1,x2,y2 = trk_obj.last_detection.data['bbox']
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
                cv2.putText(frame, f"Player {pid}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
                

        out.write(frame)
        progress_bar.update(1)

    progress_bar.close()
    cap.release()
    out.release()

    print(f"[INFO] Done. Output saved: {out_path}")
    print(f"[INFO] CSV saved: {csv_filename}")

# ============================
# 6) 실행
# ============================
if __name__ == "__main__":
    main()