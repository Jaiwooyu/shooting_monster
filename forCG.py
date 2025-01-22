import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from tqdm import tqdm
import torch

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_distance(point1, point2):
    """
    두 점 사이의 유클리디안 거리를 계산하는 함수
    """
    x1, y1 = point1
    x2, y2 = point2
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def calculate_angle(a, b, c):
    """
    세 점의 좌표를 받아 b점을 기준으로 한 각도를 계산하는 함수
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
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

def draw_custom_skeleton(frame, landmarks):
    """
    CG 느낌의 간지나는 스켈레톤을 그립니다.
    1. 먼저 두께가 큰 반투명 선(글로우 효과)을 그림
    2. 그 위에 얇은 선을 덧그려 선명하게 표시하고,
    3. 관절에는 바깥쪽 원(그림자 효과)과 안쪽 원(채움)을 그립니다.
    """
    connections = mp_pose.POSE_CONNECTIONS

    # 글로우 효과 및 내부 선 색상 설정
    glow_color = (50, 250, 250)    # 밝은 청록
    inner_color = (0, 255, 255)     # 노란 느낌
    joint_outer_color = (0, 100, 255)
    joint_inner_color = (0, 255, 255)

    # 글로우 효과 (두께가 큰 선)
    for connection in connections:
        start_idx, end_idx = connection
        if (start_idx in landmarks) and (end_idx in landmarks):
            cv2.line(frame,
                     landmarks[start_idx],
                     landmarks[end_idx],
                     glow_color, thickness=12, lineType=cv2.LINE_AA)

    # 실제 선 (얇은 선)
    for connection in connections:
        start_idx, end_idx = connection
        if (start_idx in landmarks) and (end_idx in landmarks):
            cv2.line(frame,
                     landmarks[start_idx],
                     landmarks[end_idx],
                     inner_color, thickness=4, lineType=cv2.LINE_AA)

    # 관절 표시
    for idx in landmarks:
        center = landmarks[idx]
        cv2.circle(frame, center, 10, joint_outer_color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(frame, center, 5, joint_inner_color, thickness=-1, lineType=cv2.LINE_AA)

def draw_pretty_basketball(frame, box):
    """
    농구공의 위치를 box 정보를 이용하여, 
    매우 동글고 아름다운 테두리 효과(여러 겹의 원, 글로우 효과)를 적용해 표시합니다.
    """
    bx1, by1, bx2, by2 = box
    center_x = (bx1 + bx2) // 2
    center_y = (by1 + by2) // 2
    # box의 크기를 참고하여 반지름 설정 (원본 box의 가로/세로 길이의 평균의 0.5배 정도)
    radius = int(0.5 * ((bx2 - bx1) + (by2 - by1)) / 2)

    # 외부 글로우 원 (반투명, 두께가 큰 원)
    overlay = frame.copy()
    cv2.circle(overlay, (center_x, center_y), int(radius*1.4), (50, 250, 250), thickness=20, lineType=cv2.LINE_AA)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 중간 테두리 (약간 더 두꺼운 선)
    cv2.circle(frame, (center_x, center_y), int(radius*1.2), (0, 255, 255), thickness=8, lineType=cv2.LINE_AA)
    # 내부 테두리 (얇은 선)
    cv2.circle(frame, (center_x, center_y), radius, (0, 200, 255), thickness=3, lineType=cv2.LINE_AA)
    # 중심 채움 (작은 원)
    cv2.circle(frame, (center_x, center_y), int(radius*0.3), (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    # 텍스트 추가 (선택사항)
    cv2.putText(frame, "Basketball", (bx1, by1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

def main():
    # YOLO 모델 로드 - 사람과 농구공 각각 탐지
    person_model = YOLO('yolov8l.pt')
    ball_model = YOLO('best_2.pt')
    
    person_model.to(device)
    ball_model.to(device)

    pose_estimator = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture('dataset1/cg.mp4')
    if not cap.isOpened():
        print("Error: Cannot open input video")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('dataset1/kbl_output.mp4', fourcc, fps, (width, height))

    pbar = tqdm(total=total_frames, desc="Processing frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ------------------------------
        # 농구공 탐지 처리
        # ------------------------------
        ball_results = ball_model.predict(frame, imgsz=1280, conf=0.3, verbose=False)
        ball_det = ball_results[0].boxes

        ball_boxes = ball_det.xyxy.cpu().numpy() if torch.cuda.is_available() else ball_det.xyxy.numpy()
        ball_confs = ball_det.conf.cpu().numpy() if torch.cuda.is_available() else ball_det.conf.numpy()
        ball_box = None
        max_ball_area = 0

        for j in range(len(ball_boxes)):
            x1, y1, x2, y2 = ball_boxes[j]
            area = (x2 - x1) * (y2 - y1)
            if area > max_ball_area:
                max_ball_area = area
                ball_box = (int(x1), int(y1), int(x2), int(y2))
        
        if ball_box is not None:
            # 아름다운 농구공 표시는 별도의 함수 호출
            draw_pretty_basketball(frame, ball_box)

        # ------------------------------
        # 사람 탐지 처리
        # ------------------------------
        person_results = person_model.predict(frame, imgsz=640, conf=0.4, verbose=False)
        det = person_results[0].boxes
        boxes = det.xyxy.cpu().numpy() if torch.cuda.is_available() else det.xyxy.numpy()
        cls = det.cls.cpu().numpy() if torch.cuda.is_available() else det.cls.numpy()

        # 프론트맨(가장 큰 박스)를 선택
        front_person_box = None
        max_area = 0
        for i in range(len(boxes)):
            if int(cls[i]) != 0:
                continue  # 사람 클래스만 처리
            x1, y1, x2, y2 = map(int, boxes[i])
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                front_person_box = (x1, y1, x2, y2)

        if front_person_box is not None:
            x1, y1, x2, y2 = front_person_box
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            # 사람 ROI 추출 및 MediaPipe Pose 적용
            person_roi = frame[y1:y2, x1:x2].copy()
            roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            pose_results = pose_estimator.process(roi_rgb)
            if pose_results.pose_landmarks:
                landmarks_px = {}
                for idx, lm in enumerate(pose_results.pose_landmarks.landmark):
                    cx = int(lm.x * (x2 - x1)) + x1
                    cy = int(lm.y * (y2 - y1)) + y1
                    landmarks_px[idx] = (cx, cy)
                
                # CG 스타일 스켈레톤 그리기
                draw_custom_skeleton(frame, landmarks_px)
                
                # 예시로 팔꿈치 각도를 계산하여 텍스트 표기 (필요에 따라 추가)
                try:
                    r_shoulder = landmarks_px[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    r_elbow = landmarks_px[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                    r_wrist = landmarks_px[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                    angle_r_elbow = calculate_angle(r_shoulder, r_elbow, r_wrist)
                    
                    l_shoulder = landmarks_px[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    l_elbow = landmarks_px[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                    l_wrist = landmarks_px[mp_pose.PoseLandmark.LEFT_WRIST.value]
                    angle_l_elbow = calculate_angle(l_shoulder, l_elbow, l_wrist)
                    
                    cv2.putText(frame, f'R_Elbow:{int(angle_r_elbow)}', r_elbow,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f'L_Elbow:{int(angle_l_elbow)}', l_elbow,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                except Exception as e:
                    pass

            # 선택 사항: 프론트맨의 bbox를 그림
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Player", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        out.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nProcessing complete!")
    print("Output video saved as 'dataset1/kbl_output.mp4'")

if __name__ == "__main__":
    main()
