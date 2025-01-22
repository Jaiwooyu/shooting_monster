import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from tqdm import tqdm
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def preprocess_frame(frame):
    # 영상 전처리 (필요시 추가)
    return frame

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360.0 - angle

def calculate_trunk_angle(shoulder_mid, hip_mid):
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    angle_deg = np.degrees(np.arctan2(dy, dx))
    return angle_deg - 90

def fx(x, dt):
    """ 상태 전이 함수 """
    # 상태 벡터: [x, y, vx, vy, ax, ay]
    F = np.array([
        [1, 0, dt, 0, 0.5*dt**2, 0],
        [0, 1, 0, dt, 0, 0.5*dt**2],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    return np.dot(F, x)

def hx(x):
    """ 관측 함수 """
    # 관측 벡터: [x, y]
    return x[:2]

def initialize_ukf(dt=1.0):
    points = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2., kappa=0)
    ukf = UKF(dim_x=6, dim_z=2, fx=fx, hx=hx, dt=dt, points=points)
    
    # 초기 상태 추정
    ukf.x = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
    
    # 초기 공분산 행렬
    ukf.P *= 500.
    
    # 프로세스 노이즈 공분산
    ukf.Q = np.eye(6) * 0.01
    
    # 측정 노이즈 공분산
    ukf.R = np.eye(2) * 10.
    
    return ukf

def main():
    # YOLO 모델 초기화
    person_model = YOLO('yolov8n.pt')  # 사람 감지 모델
    ball_model = YOLO('best.pt')       # 농구공 감지 모델
    
    # MediaPipe Pose 초기화
    pose_estimator = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # UKF 초기화
    ukf = initialize_ukf(dt=1.0)
    initialized = False
    
    # 비디오 캡처 초기화
    cap = cv2.VideoCapture('dataset1/input_video.mp4')
    if not cap.isOpened():
        print("Error: Cannot open input_video.mp4")
        return
    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 이미지 크기를 32의 배수로 조정
    adj_width = int(np.ceil(width / 32) * 32)
    adj_height = int(np.ceil(height / 32) * 32)
    
    # 비디오 라이터 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_with_ukf.mp4', fourcc, fps, (width, height))
    
    # 로딩바 초기화
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 전처리
        frame = preprocess_frame(frame)
        
        # YOLO를 이용한 농구공 검출
        results = ball_model.predict(frame, imgsz=[adj_width, adj_height], conf=0.1, verbose=False)
        ball_boxes = results[0].boxes.xyxy.cpu().numpy()
        
        if len(ball_boxes) > 0:
            # 가장 큰 박스 선택 (농구공이 가장 큰 물체라고 가정)
            max_area = 0
            best_box = None
            for box in ball_boxes:
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_box = box
            
            if best_box is not None:
                cx = (best_box[0] + best_box[2]) / 2
                cy = (best_box[1] + best_box[3]) / 2
                
                measurement = np.array([cx, cy])
                
                if not initialized:
                    ukf.x = np.array([cx, cy, 0, 0, 0, 0], dtype=np.float32)
                    initialized = True
                
                # UKF 보정
                ukf.predict()
                ukf.update(measurement)
        
        else:
            # 농구공 검출 실패 시 예측만 수행
            ukf.predict()
        
        # 예측된 농구공 위치
        pred_x, pred_y = int(ukf.x[0]), int(ukf.x[1])
        
        # 사람 검출 및 스켈레톤 처리
        results_person = person_model.predict(frame, imgsz=[adj_width, adj_height], conf=0.4, verbose=False)
        det = results_person[0].boxes
        
        boxes = det.xyxy.cpu().numpy()
        cls   = det.cls.cpu().numpy()
        
        person_box = None
        max_area = 0
        for i in range(len(boxes)):
            if int(cls[i]) == 0:  # COCO 데이터셋에서 사람 클래스는 0
                x1, y1, x2, y2 = boxes[i]
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    person_box = (int(x1), int(y1), int(x2), int(y2))
        
        if person_box is not None:
            px1, py1, px2, py2 = person_box
            px1 = max(px1, 0)
            py1 = max(py1, 0)
            px2 = min(px2, width)
            py2 = min(py2, height)
            
            person_roi = frame[py1:py2, px1:px2].copy()
            roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            pose_results = pose_estimator.process(roi_rgb)
            
            if pose_results.pose_landmarks:
                landmarks_px = {}
                for idx, lm in enumerate(pose_results.pose_landmarks.landmark):
                    cx_land = int(lm.x * (px2 - px1)) + px1
                    cy_land = int(lm.y * (py2 - py1)) + py1
                    landmarks_px[idx] = (cx_land, cy_land)
                
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
            
            # 예측된 농구공 위치 표시
            cv2.circle(frame, (pred_x, pred_y), 15, (0, 255, 255), 2)
            cv2.putText(frame, "Predicted Basketball", (pred_x - 50, pred_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            
            # 결과 프레임 저장
            out.write(frame)
            
            # 로딩바 업데이트
            pbar.update(1)
    
        # 리소스 해제
        pbar.close()
        cap.release()
        out.release()
        pose_estimator.close()
        cv2.destroyAllWindows()
    
        print("Processing complete! Output saved as 'output_with_ukf.mp4'.")
    
    if __name__ == "__main__":
        main()
