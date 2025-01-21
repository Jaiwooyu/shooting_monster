import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from tqdm import tqdm

mp_pose = mp.solutions.pose

def preprocess_frame(frame):
    # 필요에 따라 전처리 적용
    return frame

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360.0 - angle

def calculate_trunk_angle(shoulder_mid, hip_mid):
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    angle_deg = np.degrees(np.arctan2(dy, dx))
    return angle_deg - 90

def initialize_kalman_6d():
    # 6차원 상태: [x, y, vx, vy, ax, ay], 측정: [x, y]
    kf = cv2.KalmanFilter(6, 2)
    # 측정 행렬
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0]
    ], np.float32)
    # 전이 행렬 (constant acceleration 모델)
    dt = 1.0  # 시간 간격 (프레임 단위)
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0, 0.5*dt*dt, 0],
        [0, 1, 0, dt, 0, 0.5*dt*dt],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ], np.float32)
    # 프로세스 및 측정 노이즈 공분산 (튜닝 필요)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    return kf

def main():
    person_model = YOLO('yolov8n.pt')
    ball_model = YOLO('best.pt')
    pose_estimator = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    kf = initialize_kalman_6d()
    measurement = np.zeros((2,1), np.float32)
    initialized = False

    cap = cv2.VideoCapture('dataset1/multi2.mov')
    if not cap.isOpened():
        print("Error: Cannot open input_video.mp4")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_with_kalman_6d.mp4', fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Processing frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = preprocess_frame(frame)

        # 높은 해상도로 YOLO 추론, imgsz를 32의 배수로 설정
        results = ball_model.predict(frame, imgsz=[1920, 1088], conf=0.1, verbose=False)
        ball_boxes = results[0].boxes.xyxy.cpu().numpy()

        if len(ball_boxes) > 0:
            max_area = 0
            best_box = None
            for box in ball_boxes:
                x1, y1, x2, y2 = box
                area = (x2-x1)*(y2-y1)
                if area > max_area:
                    max_area = area
                    best_box = box

            if best_box is not None:
                cx = (best_box[0] + best_box[2]) / 2
                cy = (best_box[1] + best_box[3]) / 2

                measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
                if not initialized:
                    # 6차원 상태 초기화
                    kf.statePre = np.array([[cx], [cy], [0], [0], [0], [0]], np.float32)
                    initialized = True

                kf.correct(measurement)

        predicted = kf.predict()
        pred_x, pred_y = int(predicted[0]), int(predicted[1])

        # 사람 검출 및 스켈레톤 처리
        results_person = person_model.predict(frame, imgsz=[1920, 1088], conf=0.4, verbose=False)
        det = results_person[0].boxes

        boxes = det.xyxy.cpu().numpy()
        cls   = det.cls.cpu().numpy()

        person_box = None
        max_area = 0
        for i in range(len(boxes)):
            if int(cls[i]) == 0:
                x1, y1, x2, y2 = boxes[i]
                area = (x2-x1)*(y2-y1)
                if area > max_area:
                    max_area = area
                    person_box = (int(x1), int(y1), int(x2), int(y2))

        if person_box is not None:
            px1, py1, px2, py2 = person_box
            px1 = max(px1, 0); py1 = max(py1, 0)
            px2 = min(px2, width); py2 = min(py2, height)

            person_roi = frame[py1:py2, px1:px2].copy()
            roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            pose_results = pose_estimator.process(roi_rgb)

            if pose_results.pose_landmarks:
                landmarks_px = {}
                for idx, lm in enumerate(pose_results.pose_landmarks.landmark):
                    cx_land = int(lm.x * (px2 - px1)) + px1
                    cy_land = int(lm.y * (py2 - py1)) + py1
                    landmarks_px[idx] = (cx_land, cy_land)

                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    if start_idx in landmarks_px and end_idx in landmarks_px:
                        cv2.line(frame, landmarks_px[start_idx], landmarks_px[end_idx], (0, 255, 0), 2)

                for idx in landmarks_px:
                    cv2.circle(frame, landmarks_px[idx], 4, (0, 0, 255), -1)

                # 각도 계산 및 텍스트 표시 생략 (필요시 동일하게 추가)

            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
            cv2.putText(frame, "Player", (px1, py1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        # 예측된 농구공 위치 표시
        cv2.circle(frame, (pred_x, pred_y), 20, (0,255,255), 2)
        cv2.putText(frame, "Predicted Basketball", (pred_x, pred_y-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        out.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    pose_estimator.close()

if __name__ == "__main__":
    main()
