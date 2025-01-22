import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from tqdm import tqdm

mp_pose = mp.solutions.pose

def preprocess_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    frame_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return frame_eq

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def calculate_trunk_angle(shoulder_mid, hip_mid):
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    angle_deg = np.degrees(np.arctan2(dy, dx))
    trunk_angle = angle_deg - 90
    return trunk_angle

def main():
    person_model = YOLO('yolov8n.pt')
    ball_model = YOLO('best.pt')

    # MediaPipe Pose 초기화
    pose_estimator = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # OpenCV CSRT 추적기 초기화
    tracker = cv2.legacy.TrackerCSRT_create()  # 수정 코드

    tracking_initialized = False

    cap = cv2.VideoCapture('dataset1/multi.mov')
    if not cap.isOpened():
        print("Error: Cannot open input_video.mp4")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test3output.mp4', fourcc, fps, (width, height))

    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 이미지 전처리
        frame = preprocess_frame(frame)

        # 농구공 검출 또는 추적기 업데이트
        if frame_idx % 10 == 0 or not tracking_initialized:
            # imgsz를 32의 배수로 맞춤 (예: width와 height를 가장 가까운 32 배수로 조정)
            new_width = int((width*2) // 32 * 32)
            new_height = int((height*2) // 32 * 32)

            ball_results = ball_model.predict(frame, imgsz=[new_width, new_height], conf=0.1, verbose=False)
            ball_det = ball_results[0].boxes
            ball_boxes = ball_det.xyxy.cpu().numpy()  

            ball_box = None
            max_area = 0
            for box in ball_boxes:
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    ball_box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))  # CSRT expects (x,y,w,h)
            
            if ball_box is not None:
                tracker = cv2.legacy.TrackerCSRT_create()  # 수정 코드
                tracker.init(frame, ball_box)
                tracking_initialized = True
        else:
            tracking_success, ball_box = tracker.update(frame)
            if not tracking_success:
                tracking_initialized = False

        # 농구공 표시
        if tracking_initialized and ball_box is not None:
            x, y, w, h = ball_box
            center_x = int(x + w/2)
            center_y = int(y + h/2)
            radius = int(max(w, h) / 2)
            cv2.circle(frame, (center_x, center_y), radius, (0, 255, 255), 2)
            cv2.putText(frame, "Basketball", (int(x), int(y)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # 사람 탐지 및 스켈레톤 그리기
        person_results = person_model.predict(frame, imgsz=[width, height], conf=0.4, verbose=False)
        det = person_results[0].boxes

        boxes = det.xyxy.cpu().numpy()
        cls   = det.cls.cpu().numpy()

        person_box = None
        max_area = 0
        for i in range(len(boxes)):
            if int(cls[i]) == 0:  
                x1, y1, x2, y2 = boxes[i]
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    person_box = (int(x1), int(y1), int(x2), int(y2))

        if person_box is not None:
            px1, py1, px2, py2 = person_box
            px1 = max(0, px1); py1 = max(0, py1)
            px2 = min(width, px2); py2 = min(height, py2)

            person_roi = frame[py1:py2, px1:px2].copy()
            roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            pose_results = pose_estimator.process(roi_rgb)

            if pose_results.pose_landmarks:
                landmarks_px = {}
                for idx, lm in enumerate(pose_results.pose_landmarks.landmark):
                    cx = int(lm.x * (px2 - px1)) + px1
                    cy = int(lm.y * (py2 - py1)) + py1
                    landmarks_px[idx] = (cx, cy)

                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    if (start_idx in landmarks_px) and (end_idx in landmarks_px):
                        cv2.line(frame,
                                 landmarks_px[start_idx],
                                 landmarks_px[end_idx],
                                 (0, 255, 0), 2)

                for idx in landmarks_px:
                    cv2.circle(frame, landmarks_px[idx], 4, (0, 0, 255), -1)

                # 각도 계산 및 표시
                r_shoulder = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
                r_elbow    = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
                r_wrist    = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_WRIST.value)
                l_shoulder = landmarks_px.get(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
                l_elbow    = landmarks_px.get(mp_pose.PoseLandmark.LEFT_ELBOW.value)
                l_wrist    = landmarks_px.get(mp_pose.PoseLandmark.LEFT_WRIST.value)
                r_hip      = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_HIP.value)
                r_knee     = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_KNEE.value)
                r_ankle    = landmarks_px.get(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
                l_hip      = landmarks_px.get(mp_pose.PoseLandmark.LEFT_HIP.value)
                l_knee     = landmarks_px.get(mp_pose.PoseLandmark.LEFT_KNEE.value)
                l_ankle    = landmarks_px.get(mp_pose.PoseLandmark.LEFT_ANKLE.value)

                if r_shoulder and r_elbow and r_wrist:
                    angle_r_elbow = calculate_angle(r_shoulder, r_elbow, r_wrist)
                    cv2.putText(frame, f'R_Elbow:{int(angle_r_elbow)}',
                                r_elbow, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                if l_shoulder and l_elbow and l_wrist:
                    angle_l_elbow = calculate_angle(l_shoulder, l_elbow, l_wrist)
                    cv2.putText(frame, f'L_Elbow:{int(angle_l_elbow)}',
                                l_elbow, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                if r_hip and r_knee and r_ankle:
                    angle_r_knee = calculate_angle(r_hip, r_knee, r_ankle)
                    cv2.putText(frame, f'R_Knee:{int(angle_r_knee)}',
                                r_knee, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                if l_hip and l_knee and l_ankle:
                    angle_l_knee = calculate_angle(l_hip, l_knee, l_ankle)
                    cv2.putText(frame, f'L_Knee:{int(angle_l_knee)}',
                                l_knee, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                if r_shoulder and l_shoulder and r_hip and l_hip:
                    shoulder_mid = ((r_shoulder[0] + l_shoulder[0]) // 2,
                                    (r_shoulder[1] + l_shoulder[1]) // 2)
                    hip_mid = ((r_hip[0] + l_hip[0]) // 2,
                               (r_hip[1] + l_hip[1]) // 2)
                    trunk_angle = calculate_trunk_angle(shoulder_mid, hip_mid)
                    cv2.putText(frame, f'Trunk:{int(trunk_angle)}',
                                shoulder_mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
            cv2.putText(frame, "Player", (px1, py1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        out.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    pose_estimator.close()

    print("Processing complete! Output saved as 'test3output.mp4'.")

if __name__ == "__main__":
    main()
