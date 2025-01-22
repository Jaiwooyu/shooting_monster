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
    
    Parameters:
        point1 (tuple): 첫 번째 점의 (x, y) 좌표
        point2 (tuple): 두 번째 점의 (x, y) 좌표
        
    Returns:
        float: 두 점 사이의 유클리디안 거리
    """
    x1, y1 = point1
    x2, y2 = point2
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def calculate_angle(a, b, c):
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

class BallTracker:
    def __init__(self):
        self.prev_ball_center = None
        self.prev_ball_radius = None
        self.frames_since_detection = 0
        self.max_frames_to_keep = 10  # 이전 위치를 유지할 최대 프레임 수
        
    def update(self, ball_detections):
        current_ball = None
        max_ball_area = 0
        
        # 현재 프레임의 농구공 검출 결과 처리
        if len(ball_detections) > 0:
            for box in ball_detections:
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                if area > max_ball_area:
                    max_ball_area = area
                    current_ball = (int(x1), int(y1), int(x2), int(y2))
                    
            if current_ball is not None:
                x1, y1, x2, y2 = current_ball
                center_x = int((x1 + x2) // 2)
                center_y = int((y1 + y2) // 2)
                radius = int(0.5 * max(x2 - x1, y2 - y1) / 2)
                
                # 반환할 center 좌표를 정수형 튜플로 생성
                ball_center = (int(center_x), int(center_y))
                
                # 이전 위치가 있는 경우, 급격한 위치 변화 확인
                if self.prev_ball_center is not None:
                    prev_x, prev_y = self.prev_ball_center
                    distance = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                    
                    # 급격한 위치 변화가 있는 경우 이전 위치 유지
                    if distance > radius * 4:  # 임계값은 조정 가능
                        return self.prev_ball_center, self.prev_ball_radius
                
                self.prev_ball_center = ball_center
                self.prev_ball_radius = radius
                self.frames_since_detection = 0
                return (center_x, center_y), radius
                
        # 현재 프레임에서 농구공이 검출되지 않은 경우
        if self.prev_ball_center is not None and self.frames_since_detection < self.max_frames_to_keep:
            self.frames_since_detection += 1
            return self.prev_ball_center, self.prev_ball_radius
            
        return None, None

def main():
    # YOLO 모델 GPU 설정
    person_model = YOLO('yolov8l.pt')
    ball_model = YOLO('best_2.pt')
    print(ball_model.names)
    
    # 모델을 GPU로 이동
    person_model.to(device)
    ball_model.to(device)

    pose_estimator = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # 볼 트래커 초기화
    ball_tracker = BallTracker()

    cap = cv2.VideoCapture('dataset1/kbl_input.mov')
    if not cap.isOpened():
        print("Error: Cannot open input video")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # CUDA 가속을 위한 VideoWriter 설정
    if torch.cuda.is_available():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('dataset1/kbl_output.mp4', fourcc, fps, (width, height))
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('dataset1/kbl_output.mp4', fourcc, fps, (width, height))

    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 이미지 전처리 및 GPU로 전송
        if torch.cuda.is_available():
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_chw = frame_rgb.transpose(2, 0, 1)
            frame_cuda = torch.from_numpy(frame_chw).unsqueeze(0).to(device)
        else:
            frame_cuda = frame

        # 농구공 탐지 및 추적
        ball_results = ball_model.predict(frame, imgsz=1280, conf=0.2, verbose=False, classes=[0])
        ball_det = ball_results[0].boxes
        ball_boxes = ball_det.xyxy.cpu().numpy() if torch.cuda.is_available() else ball_det.xyxy.numpy()
        
        # 트래커 업데이트
        ball_center, ball_radius = ball_tracker.update(ball_boxes)

        # 농구공 표시
        if ball_center is not None and ball_radius is not None:
            cv2.circle(frame, ball_center, ball_radius, (0, 255, 255), 2)
            cv2.putText(frame, "Basketball", 
                      (ball_center[0] - ball_radius, ball_center[1] - ball_radius - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # 사람 탐지
        person_results = person_model.predict(frame, imgsz=640, conf=0.4, verbose=False)
        det = person_results[0].boxes
        boxes = det.xyxy.cpu().numpy() if torch.cuda.is_available() else det.xyxy.numpy()
        cls = det.cls.cpu().numpy() if torch.cuda.is_available() else det.cls.numpy()

        closest_person_box = None
        min_distance = float('inf')
        
        if ball_center is not None:
            for i in range(len(boxes)):
                if int(cls[i]) == 0:  # person class
                    x1, y1, x2, y2 = map(int, boxes[i])
                    person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    distance = calculate_distance(ball_center, person_center)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_person_box = (x1, y1, x2, y2)

        # MediaPipe Pose 처리
        if closest_person_box is not None:
            px1, py1, px2, py2 = closest_person_box
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

                # 스켈레톤 그리기
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
                r_shoulder = landmarks_px[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                r_elbow = landmarks_px[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                r_wrist = landmarks_px[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                angle_r_elbow = calculate_angle(r_shoulder, r_elbow, r_wrist)

                l_shoulder = landmarks_px[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                l_elbow = landmarks_px[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                l_wrist = landmarks_px[mp_pose.PoseLandmark.LEFT_WRIST.value]
                angle_l_elbow = calculate_angle(l_shoulder, l_elbow, l_wrist)

                r_hip = landmarks_px[mp_pose.PoseLandmark.RIGHT_HIP.value]
                r_knee = landmarks_px[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                r_ankle = landmarks_px[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                angle_r_knee = calculate_angle(r_hip, r_knee, r_ankle)

                l_hip = landmarks_px[mp_pose.PoseLandmark.LEFT_HIP.value]
                l_knee = landmarks_px[mp_pose.PoseLandmark.LEFT_KNEE.value]
                l_ankle = landmarks_px[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                angle_l_knee = calculate_angle(l_hip, l_knee, l_ankle)

                shoulder_mid = ((r_shoulder[0] + l_shoulder[0]) // 2,
                              (r_shoulder[1] + l_shoulder[1]) // 2)
                hip_mid = ((r_hip[0] + l_hip[0]) // 2,
                          (r_hip[1] + l_hip[1]) // 2)
                trunk_angle = calculate_trunk_angle(shoulder_mid, hip_mid)

                # 각도 텍스트 표시
                cv2.putText(frame, f'R_Elbow:{int(angle_r_elbow)}',
                          r_elbow, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame, f'L_Elbow:{int(angle_l_elbow)}',
                          l_elbow, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame, f'R_Knee:{int(angle_r_knee)}',
                          r_knee, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame, f'L_Knee:{int(angle_l_knee)}',
                          l_knee, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame, f'Trunk:{int(trunk_angle)}',
                          shoulder_mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
            cv2.putText(frame, "Player", (px1, py1 - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        out.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nProcessing complete!")
    print("Output video saved as 'kbl_output.mp4'")

if __name__ == "__main__":
    main()