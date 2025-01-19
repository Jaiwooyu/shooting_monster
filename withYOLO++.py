import cv2
import csv
import math
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from norfair import Detection, Tracker, draw_tracked_objects
from tqdm import tqdm  # tqdm 임포트

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

    # 예시 레퍼런스 각도
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
    model = YOLO("yolov8s.pt")
    
    # --------------------------------
    # (B) 입력 비디오 & 결과 비디오 설정
    # --------------------------------
    input_video = "dataset/multi.mp4"
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error opening video: {input_video}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 전체 프레임 수

    out_path = "dataset/output_mul.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    print(f"[INFO] Video output: {out_path}")

    # --------------------------------
    # (C) CSV 로깅 (플레이어별 데이터)
    # --------------------------------
    csv_filename = "dataset/basketball_game_data.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "player_id", "ball_state",  # ball_state: 'holding', 'dribbling', or 'none'
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
        results = model.predict(
            source=frame,
            conf=MIN_CONF,
            iou=MIN_IOU,
            device=0,
            verbose=False
        )
        detections = results[0]
        
        person_detections = []
        ball_detections   = []

        if detections.boxes is not None:
            for box in detections.boxes:
                cls_id = int(box.cls[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # (a) person
                if cls_id == 0: 
                    cx, cy = get_bbox_center([x1, y1, x2, y2])
                    person_detections.append(
                        Detection(points=np.array([[cx, cy]]),
                                  data={'bbox':[x1, y1, x2, y2]})
                    )
                # (b) sports ball
                elif cls_id == 32:
                    cx, cy = get_bbox_center([x1, y1, x2, y2])
                    ball_detections.append(
                        Detection(points=np.array([[cx, cy]]),
                                  data={'bbox':[x1, y1, x2, y2]})
                    )

        # -------------------------------
        # 2) Norfair로 사람/공 각각 추적
        #    -> player_tracker, ball_tracker 업데이트
        # -------------------------------
        tracked_players = player_tracker.update(person_detections)
        tracked_balls   = ball_tracker.update(ball_detections)

        # (선택) 시각화를 위해 bounding box, id 표시
        for trk_obj in tracked_players:
            pid = trk_obj.global_id
            cx, cy = trk_obj.estimate[0]
            if trk_obj.last_detection is not None:
                x1,y1,x2,y2 = trk_obj.last_detection.data['bbox']
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
                cv2.putText(frame, f"Player {pid}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        
        for trk_obj in tracked_balls:
            bid = trk_obj.global_id
            if trk_obj.last_detection is not None:
                x1,y1,x2,y2 = trk_obj.last_detection.data['bbox']
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,255), 2)
                cv2.putText(frame, f"Ball {bid}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        # -------------------------------
        # 3) "공을 들고 있는/드리블 중인" 플레이어 식별
        # -------------------------------
        player_pose_data = {}  
        row_to_save = []
        
        for trk_obj in tracked_players:
            pid = trk_obj.global_id
            if trk_obj.last_detection is None:
                continue
            x1p, y1p, x2p, y2p = trk_obj.last_detection.data['bbox']
            x1p = max(0, int(x1p)); y1p = max(0, int(y1p))
            x2p = min(width,  int(x2p)); y2p = min(height, int(y2p))
            if (x2p - x1p < 10) or (y2p - y1p < 10):
                continue

            person_roi = frame_rgb[y1p:y2p, x1p:x2p]
            results_pose = pose_detector.process(person_roi)

            angle_arm = 0.0
            angle_leg = 0.0
            shot_score = 0.0
            landmarks_px = {}

            if results_pose.pose_landmarks:
                hR, wR = person_roi.shape[:2]
                for idx, lm in enumerate(results_pose.pose_landmarks.landmark):
                    cx = int(lm.x * wR) + x1p
                    cy = int(lm.y * hR) + y1p
                    landmarks_px[idx] = (cx, cy)
                
                r_shoulder = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
                r_elbow    = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
                r_wrist    = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_WRIST.value)
                r_hip      = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_HIP.value)
                r_knee     = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_KNEE.value)
                r_ankle    = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

                if (r_shoulder and r_elbow and r_wrist):
                    angle_arm = calculate_angle(r_shoulder, r_elbow, r_wrist)
                if (r_hip and r_knee and r_ankle):
                    angle_leg = calculate_angle(r_hip, r_knee, r_ankle)
                
                shot_score = score_shot_form(landmarks_px)

                for conn in mp_pose.POSE_CONNECTIONS:
                    start, end = conn
                    if (start in landmarks_px) and (end in landmarks_px):
                        cv2.line(frame, landmarks_px[start], landmarks_px[end], (0,255,0), 2)
                for idx_lm in landmarks_px:
                    cv2.circle(frame, landmarks_px[idx_lm], 4, (0,0,255), -1)
                
                if r_elbow:
                    cv2.putText(frame, f"Arm:{angle_arm:.1f}",
                                (r_elbow[0]+10, r_elbow[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                if r_knee:
                    cv2.putText(frame, f"Leg:{angle_leg:.1f}",
                                (r_knee[0]+10, r_knee[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                if r_shoulder:
                    cv2.putText(frame, f"Score:{shot_score:.1f}",
                                (r_shoulder[0]+20, r_shoulder[1]-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            player_pose_data[pid] = {
                "landmarks": landmarks_px,
                "bbox": (x1p, y1p, x2p, y2p),
                "angle_arm": angle_arm,
                "angle_leg": angle_leg,
                "shot_score": shot_score
            }

        ball_positions = {}
        for trk_obj in tracked_balls:
            bid = trk_obj.global_id
            if trk_obj.last_detection is not None:
                x1b,y1b,x2b,y2b = trk_obj.last_detection.data['bbox']
                cx_b, cy_b = get_bbox_center([x1b,y1b,x2b,y2b])
                ball_positions[bid] = (cx_b, cy_b, (x1b,y1b,x2b,y2b))

        for bid, (cx_b, cy_b, bb_ball) in ball_positions.items():
            if bid not in ball_movement_history:
                ball_movement_history[bid] = []

            min_dist = 999999
            nearest_pid = None
            nearest_hand = None

            for pid, pdata in player_pose_data.items():
                lms = pdata["landmarks"]
                if not lms:
                    continue
                left_wrist  = lms.get(mp_pose.PoseLandmark.LEFT_WRIST.value)
                right_wrist = lms.get(mp_pose.PoseLandmark.RIGHT_WRIST.value)
                left_ankle  = lms.get(mp_pose.PoseLandmark.LEFT_ANKLE.value)
                right_ankle = lms.get(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

                cand_points = []
                if left_wrist:  cand_points.append(("left", left_wrist))
                if right_wrist: cand_points.append(("right", right_wrist))
                for hand_id, (hx, hy) in cand_points:
                    dist = np.hypot(hx - cx_b, hy - cy_b)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_pid = pid
                        nearest_hand = hand_id

                foot_near = False
                for foot_coord in [left_ankle, right_ankle]:
                    if foot_coord:
                        fd = np.hypot(foot_coord[0] - cx_b, foot_coord[1] - cy_b)
                        if fd < DRIBBLE_FOOT_THRESHOLD:
                            foot_near = True
                            break

                if foot_near:
                    if len(ball_movement_history[bid]) == 0 or \
                       ball_movement_history[bid][-1][0] != "down":
                        ball_movement_history[bid].append(("down", frame_idx))

            ball_state = "none"
            if nearest_pid is not None and min_dist < BALL_HAND_THRESHOLD:
                ball_state = "holding"
                if len(ball_movement_history[bid]) == 0 or \
                   ball_movement_history[bid][-1][0] != "up":
                    ball_movement_history[bid].append(("up", frame_idx))

                recent_actions = ball_movement_history[bid]
                if len(recent_actions) >= 2:
                    last_action = recent_actions[-1]
                    prev_action = recent_actions[-2]
                    if last_action[0] == "up" and prev_action[0] == "down":
                        if (last_action[1] - prev_action[1]) < DRIBBLE_TIME_WINDOW:
                            ball_state = "dribbling"

            if ball_state != "none" and nearest_pid is not None:
                cv2.putText(frame, ball_state.upper(), (int(cx_b)+10, int(cy_b)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                head = player_pose_data[nearest_pid]["landmarks"].get(mp_pose.PoseLandmark.NOSE.value)
                if head:
                    cv2.putText(frame, ball_state.upper(), (head[0], head[1]-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                row_to_save.append([
                    frame_idx, 
                    nearest_pid,
                    ball_state,
                    round(player_pose_data[nearest_pid]["angle_arm"],1),
                    round(player_pose_data[nearest_pid]["angle_leg"],1),
                    round(player_pose_data[nearest_pid]["shot_score"],1)
                ])

        recorded_pids = [r[1] for r in row_to_save]
        for pid, pdata in player_pose_data.items():
            if pid not in recorded_pids:
                row_to_save.append([
                    frame_idx,
                    pid,
                    "none",
                    round(pdata["angle_arm"],1),
                    round(pdata["angle_leg"],1),
                    round(pdata["shot_score"],1)
                ])

        if len(row_to_save) > 0:
            with open(csv_filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for row_ in row_to_save:
                    writer.writerow(row_)

        out.write(frame)
        
        progress_bar.update(1)  # tqdm 진행률 업데이트

    progress_bar.close()  # 로딩바 종료
    cap.release()
    out.release()
    # cv2.destroyAllWindows()

    print(f"[INFO] Done. Output saved: {out_path}")
    print(f"[INFO] CSV saved: {csv_filename}")

# ============================
# 6) 실행
# ============================
if __name__ == "__main__":
    main()
