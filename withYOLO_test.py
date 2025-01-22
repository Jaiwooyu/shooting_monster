import cv2
import csv
import math
import numpy as np
import torch
import mediapipe as mp
from ultralytics import YOLO
from norfair import Detection, Tracker, draw_tracked_objects
from tqdm import tqdm  # tqdm 임포트
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) #futurewarning 무시


# ============================
# 1) 설정 상수/파라미터
# ============================
MIN_CONF = 0.05       # YOLO confidence threshold
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

# ============================
# 4) Norfair Tracker 설정
#    - 플레이어용 Tracker, 공용 Tracker를 별도로 운영
# ============================
# (A) Norfair가 요구하는 distance function
def euclidean_distance(detection, tracked_object):
    """
    detection: Norfair Detection 객체
    tracked_object: Norfair TrackedObject 객체
    둘 다 .points (shape: Nx2) 가 있음 (여기선 Nx=1, x= [cx, cy])
    """
    det_xy = detection.points[0]
    trk_xy = tracked_object.estimate[0]
    return np.linalg.norm(det_xy - trk_xy)

# (B) Tracker 생성
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
    # (A) YOLOv8 모델 로드
    # --------------------------------
    model = YOLO("yolov8l.pt")
    ball_model = YOLO("best.pt")
    print("클래스 이름들:", ball_model.names)


    # --------------------------------
    # (B) 입력 비디오 & 결과 비디오 설정
    # --------------------------------
    input_video = "dataset1/multi.mov"
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error opening video: {input_video}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 전체 프레임 수

    out_path = "dataset1/output_mul_newmod.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    print(f"[INFO] Video output: {out_path}")

    # --------------------------------
    # (D) MediaPipe Pose 준비
    # --------------------------------
    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.5
    )

    frame_idx = 0

    # 드리블 판정을 위한 임시 저장: {ball_id: [("down", frame_idx), ("up", frame_idx), ...]}
    ball_movement_history = {}

    # tqdm 로딩바 설정
    progress_bar = tqdm(total=total_frames, desc="Processing Video")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] No more frames. Exiting...")
            break

        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # -------------------------------
        # 1) YOLOv8 추론
        # -------------------------------
        # results = model.predict(
        #     source=frame,
        #     conf=MIN_CONF,
        #     iou=MIN_IOU,
        #     device=0,
        #     verbose=False
        # )[0]

        ball_results = ball_model.predict(
            source=frame,
            conf=MIN_CONF,
            iou=MIN_IOU,
            device=0,
            verbose=False
        )[0]

        person_detections = []
        ball_detections   = []

        # if results.boxes is not None:
        #     for box in results.boxes:
        #         if int(box.cls[0].item()) == 0:  # person class
        #             x1, y1, x2, y2 = box.xyxy[0].tolist()
        #             cx, cy = get_bbox_center([x1, y1, x2, y2])
        #             person_detections.append(
        #                 Detection(points=np.array([[cx, cy]]),
        #                         data={'bbox':[x1, y1, x2, y2]})
        #             )

        if ball_results.boxes is not None:
            for box in ball_results.boxes:  # person_results.boxes -> ball_results.boxes로 수정
                # ball 클래스 번호는 학습된 모델에 따라 다를 수 있으므로 
                # best.pt 모델에서의 ball 클래스 인덱스를 사용해야 합니다
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = get_bbox_center([x1, y1, x2, y2])
                ball_detections.append(  # person_detections -> ball_detections로 수정
                    Detection(points=np.array([[cx, cy]]),
                            data={'bbox':[x1, y1, x2, y2]})
                )


        # -------------------------------
        # 2) Norfair로 사람/공 각각 추적
        #    -> player_tracker, ball_tracker 업데이트
        # -------------------------------
        # tracked_players = player_tracker.update(person_detections)
        tracked_balls   = ball_tracker.update(ball_detections)
        print(f"tracked_balls: {tracked_balls}")

        for trk_obj in tracked_balls:
                    pid = trk_obj.global_id
                    cx, cy = trk_obj.estimate[0]
                    if trk_obj.last_detection is not None:
                        x1,y1,x2,y2 = trk_obj.last_detection.data['bbox']
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
                        cv2.putText(frame, f"Player {pid}", (int(x1), int(y1)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

        # tracked_players 처리하는 부분을 찾아서 다음과 같이 수정

        # # 공의 위치를 먼저 파악
        # ball_center = None
        # for trk_obj in tracked_balls:
        #     if trk_obj.last_detection is not None:
        #         x1b,y1b,x2b,y2b = trk_obj.last_detection.data['bbox']
        #         ball_center = get_bbox_center([x1b,y1b,x2b,y2b])
        #         break  # 첫 번째 공만 사용
        
        # print(f"Frame {frame_idx}: {len(tracked_players)} players, {len(tracked_balls)} balls")
        # print(f"Ball Center: {ball_center}")

        # # 가장 가까운 플레이어 찾기
        # nearest_player = None
        # min_dist_to_ball = float('inf')

        # # nearest_player 업데이트
        # if ball_center is not None:
        #     for trk_obj in tracked_players:
        #         if trk_obj.last_detection is not None:
        #             x1p,y1p,x2p,y2p = trk_obj.last_detection.data['bbox']
        #             player_center = get_bbox_center([x1p,y1p,x2p,y2p])
        #             dist = np.hypot(player_center[0] - ball_center[0], 
        #                         player_center[1] - ball_center[1])
        #             if dist < min_dist_to_ball:
        #                 min_dist_to_ball = dist
        #                 nearest_player = trk_obj


        # # -------------------------------
        # # 3) "공을 들고 있는/드리블 중인" 플레이어 식별
        # # -------------------------------
        # player_pose_data = {}  
        # row_to_save = []
        
        # # 가장 가까운 플레이어만 포즈 분석
        # if nearest_player is not None:
        #     pid = nearest_player.global_id
        #     if nearest_player.last_detection is not None:
        #         x1p, y1p, x2p, y2p = nearest_player.last_detection.data['bbox']
        #         x1p = max(0, int(x1p)); y1p = max(0, int(y1p))
        #         x2p = min(width, int(x2p)); y2p = min(height, int(y2p))
        #         if not ((x2p - x1p < 10) or (y2p - y1p < 10)):
        #             person_roi = frame_rgb[y1p:y2p, x1p:x2p]
        #             results_pose = pose_detector.process(person_roi)

        #             angle_arm = 0.0
        #             angle_leg = 0.0
        #             shot_score = 0.0
        #             landmarks_px = {}

        #             if results_pose.pose_landmarks:
        #                 hR, wR = person_roi.shape[:2]
        #                 for idx, lm in enumerate(results_pose.pose_landmarks.landmark):
        #                     cx = int(lm.x * wR) + x1p
        #                     cy = int(lm.y * hR) + y1p
        #                     landmarks_px[idx] = (cx, cy)
                        
        #                 r_shoulder = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        #                 r_elbow = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        #                 r_wrist = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_WRIST.value)
        #                 r_hip = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_HIP.value)
        #                 r_knee = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_KNEE.value)
        #                 r_ankle = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

        #                 if (r_shoulder and r_elbow and r_wrist):
        #                     angle_arm = calculate_angle(r_shoulder, r_elbow, r_wrist)
        #                 if (r_hip and r_knee and r_ankle):
        #                     angle_leg = calculate_angle(r_hip, r_knee, r_ankle)
                        
        #                 player_pose_data[pid] = {
        #                     "landmarks": landmarks_px,
        #                     "bbox": (x1p, y1p, x2p, y2p),
        #                     "angle_arm": angle_arm,
        #                     "angle_leg": angle_leg,
        #                     "shot_score": shot_score
        #                 }

        out.write(frame)
        
        progress_bar.update(1)

    progress_bar.close()
    cap.release()
    out.release()

    print(f"[INFO] Done. Output saved: {out_path}")
    # print(f"[INFO] CSV saved: {csv_filename}")

if __name__ == "__main__":
    main()