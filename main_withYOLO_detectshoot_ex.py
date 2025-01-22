import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from tqdm import tqdm
import torch
import math

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class ShootingDetector:
    def __init__(self):
        self.ball_history = []  # 공의 위치 이력 저장
        self.shooting_threshold = {
            'vertical_velocity': 10,      # 수직 속도 임계값
            'trajectory_points': 8,       # 궤적 분석을 위한 최소 포인트 수
            'horizontal_movements': 50,  # 추가
            'min_height_change': 40,      # 최소 수직 이동 거리
            'detection_window': 15,       # 분석할 프레임 수
            'min_curve': 0.003           # 궤적 곡률 임계값
        }
        self.debug_data = {}  # 디버깅 데이터 저장

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

        # 수직 방향 분석
        vertical_movements = [recent_points[i][1] - recent_points[i-1][1] 
                            for i in range(1, len(recent_points))]
        # 수평 방향 분석
        horizontal_movements = [recent_points[i][0] - recent_points[i-1][0] 
                              for i in range(1, len(recent_points))]

        # 전체 수직 이동 거리
        total_vertical_change = recent_points[-1][1] - recent_points[0][1]

        # 속도 계산
        vertical_speeds = [abs(v) for v in vertical_movements]
        avg_vertical_speed = np.mean(vertical_speeds)

        # 가속도 계산
        vertical_accelerations = [vertical_speeds[i] - vertical_speeds[i-1] 
                                for i in range(1, len(vertical_speeds))]

        # 궤적 곡률 계산
        curvatures = []
        for i in range(1, len(recent_points)-1):
            p0, p1, p2 = recent_points[i-1], recent_points[i], recent_points[i+1]
            area = abs(0.5 * ((p0[0]*(p1[1]-p2[1]) + p1[0]*(p2[1]-p0[1]) + p2[0]*(p0[1]-p1[1]))))
            d01 = math.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)
            d12 = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            d02 = math.sqrt((p2[0]-p0[0])**2 + (p2[1]-p0[1])**2)
            if d01 * d12 * d02 != 0:
                curvature = (4 * area) / (d01 * d12 * d02)
                curvatures.append(curvature)
            else:
                curvatures.append(0)

        avg_curvature = np.mean(curvatures) if curvatures else 0

        # 수평 변화량 계산 추가 (total_vertical_change 계산 바로 다음에)
        total_vertical_change = recent_points[-1][1] - recent_points[0][1]
        total_horizontal_change = recent_points[-1][0] - recent_points[0][0]  # 추가된 라인

        # debug_data 딕셔너리에 total_horizontal_change 추가
        self.debug_data = {
            'avg_vertical_speed': avg_vertical_speed,
            'total_vertical_change': total_vertical_change,
            'total_horizontal_change': total_horizontal_change,  # 추가된 라인
            'avg_curvature': avg_curvature,
            'recent_points': recent_points,
            'vertical_speeds': vertical_speeds,
            'vertical_accelerations': vertical_accelerations
        }
        # 슈팅 판정 조건
        condition_velocity = avg_vertical_speed > self.shooting_threshold['vertical_velocity']
        condition_height = abs(total_vertical_change) > self.shooting_threshold['min_height_change']
        condition_acceleration = any(acc > self.shooting_threshold['vertical_velocity'] for acc in vertical_accelerations)
        condition_curvature = avg_curvature > self.shooting_threshold['min_curve']
        upward_movements_ratio = sum(1 for v in vertical_movements if v < 0) / len(vertical_movements) > 0.7

        # 디버깅용 조건 상태 저장
        self.debug_data['conditions'] = {
            'Speed > Threshold': condition_velocity,
            'Height > Min Change': condition_height,
            'Has Acceleration': condition_acceleration,
            'Has Curve': condition_curvature,
            'Upward Motion': upward_movements_ratio
        }

        is_shooting = (
            condition_velocity and 
            condition_height and 
            condition_acceleration and 
            condition_curvature and 
            upward_movements_ratio
        )

        # 이력 관리 (최근 N개만 유지)
        if len(self.ball_history) > self.shooting_threshold['detection_window']:
            self.ball_history.pop(0)

        return is_shooting

def visualize_shooting_data(frame, ball_center, shooting_detector):
    """슈팅 관련 데이터를 프레임에 시각화"""
    if not shooting_detector.debug_data:
        return frame

    # 데이터 표시를 위한 시작 위치
    text_x = 50
    text_y = 50
    line_height = 30

    # 기본 데이터 표시
    cv2.putText(frame, f"Avg Speed: {shooting_detector.debug_data['avg_vertical_speed']:.2f}", 
                (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    
    cv2.putText(frame, f"Vertical Change: {shooting_detector.debug_data['total_vertical_change']:.2f}", 
                (text_x, text_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    
    cv2.putText(frame, f"Horizontal Change: {shooting_detector.debug_data['total_horizontal_change']:.2f}",
                (text_x, text_y + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    
    cv2.putText(frame, f"Curvature: {shooting_detector.debug_data['avg_curvature']:.4f}", 
                (text_x, text_y + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            
    if 'conditions' in shooting_detector.debug_data:
        for i, (condition, is_met) in enumerate(shooting_detector.debug_data['conditions'].items()):
            color = (0,255,0) if is_met else (0,0,255)
            cv2.putText(frame, f'{condition}: {is_met}',
                       (text_x, text_y + line_height * (i+4)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 궤적 그리기
    if 'recent_points' in shooting_detector.debug_data:
        points = shooting_detector.debug_data['recent_points']
        for i in range(1, len(points)):
            pt1 = (int(points[i-1][0]), int(points[i-1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    return frame

def calculate_distance(point1, point2):
    """두 점 사이의 유클리디안 거리 계산"""
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_angle(a, b, c):
    """세 점으로 이루어진 각도 계산"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def calculate_trunk_angle(shoulder_mid, hip_mid):
    """몸통 각도 계산"""
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    angle_deg = np.degrees(np.arctan2(dy, dx))  
    trunk_angle = angle_deg - 90
    return trunk_angle

def main():
    # YOLO 모델 설정
    person_model = YOLO('yolov8l.pt')
    ball_model = YOLO('best_2.pt')
    person_model.to(device)
    ball_model.to(device)

    # MediaPipe Pose 설정
    pose_estimator = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 비디오 캡처 설정
    cap = cv2.VideoCapture('dataset1/kbl_input.mov')
    if not cap.isOpened():
        print("Error: Cannot open input video")
        return

    # 비디오 속성 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 출력 비디오 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('dataset1/kbl_output.mp4', fourcc, fps, (width, height))

    # 진행 상황 표시 설정
    pbar = tqdm(total=total_frames, desc="Processing frames")

    # ShootingDetector 초기화
    shooting_detector = ShootingDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 공 감지
        ball_results = ball_model.predict(frame, imgsz=1280, conf=0.2, verbose=False, classes=[0])
        ball_det = ball_results[0].boxes
        ball_boxes = ball_det.xyxy.cpu().numpy() if torch.cuda.is_available() else ball_det.xyxy.numpy()
        
        ball_box = None
        max_ball_area = 0

        for j in range(len(ball_boxes)):
            x1, y1, x2, y2 = ball_boxes[j]
            area = (x2 - x1) * (y2 - y1)
            if area > max_ball_area:
                max_ball_area = area
                ball_box = (int(x1), int(y1), int(x2), int(y2))

        # 슈팅 감지 및 시각화
        if ball_box is not None:
            is_shooting = shooting_detector.analyze_ball_trajectory(ball_box)
            frame = visualize_shooting_data(frame, ball_box, shooting_detector)

            # 슈팅 상태 표시
            status_text = "SHOOTING!" if is_shooting else "No shooting"
            status_color = (0,255,0) if is_shooting else (0,0,255)
            cv2.putText(frame, status_text, (50, height - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
            
        # 사람 탐지 - GPU 활용 (고정된 이미지 크기 사용)
        person_results = person_model.predict(frame, imgsz=640, conf=0.4, verbose=False)  # imgsz를 고정 크기로 설정
        det = person_results[0].boxes

        # boxes와 cls를 GPU에서 가져오기
        boxes = det.xyxy.cpu().numpy() if torch.cuda.is_available() else det.xyxy.numpy()
        cls = det.cls.cpu().numpy() if torch.cuda.is_available() else det.cls.numpy()

        ball_center = None
        if len(ball_boxes) > 0:
            ball_box = ball_boxes[0]  # 가장 큰 확률의 농구공 선택
            bx1, by1, bx2, by2 = map(int, ball_box)
            ball_center = ((bx1 + bx2) // 2, (by1 + by2) // 2)
            
            # 농구공 표시
            radius = int(0.5 * max(bx2 - bx1, by2 - by1) / 2)
            cv2.circle(frame, ball_center, radius, (0, 255, 255), 2)
            cv2.putText(frame, "Basketball", (bx1, by1 - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)


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

        # 나머지 처리 (사람 감지, 포즈 추정 등) ...
        # MediaPipe Pose 처리
        if closest_person_box is not None:
            px1, py1, px2, py2 = closest_person_box
            px1 = max(0, px1); py1 = max(0, py1)
            px2 = min(width, px2); py2 = min(height, py2)

            person_roi = frame[py1:py2, px1:px2].copy()
            roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            
            # MediaPipe는 CPU에서 실행되므로, 필요한 부분만 CPU로 전송
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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nProcessing complete!")
    print("Output video saved as 'kbl_output.mp4'")

if __name__ == "__main__":
    main()