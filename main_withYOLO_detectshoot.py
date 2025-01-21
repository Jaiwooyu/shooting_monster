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
        self.ball_history = []  
        self.valid_detections = []  # 정상 검출 (frame_idx, center) 저장
        self.last_valid_ball_center = None  
        self.last_valid_frame_idx = None  
        self.shooting_threshold = {
            'vertical_velocity': 10,
            'trajectory_points': 8,
            'horizontal_movements': 50,
            'min_height_change': 40,
            'detection_window': 15,
            'min_curve': 0.003,
            'max_position_jump': 100  # 최대 허용 위치 변화 (픽셀 단위)
        }
        self.debug_data = {}
        self.last_debug_data = {}  # 이전 유효 디버그 데이터 저장

    def analyze_ball_trajectory(self, ball_box, current_frame_idx):
        """공의 궤적을 분석하여 슈팅 동작 감지 및 이상치 필터링"""
        # ball_box가 None이면 분석 불가
        if ball_box is None:
            self.ball_history.append(None)
            return False

        x1, y1, x2, y2 = ball_box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        current_center = (center_x, center_y)

        # 이전 정상 검출과 비교하여 큰 변화 감지
        if self.last_valid_ball_center is not None:
            distance = math.sqrt((current_center[0] - self.last_valid_ball_center[0])**2 + 
                                 (current_center[1] - self.last_valid_ball_center[1])**2)
            if distance > self.shooting_threshold['max_position_jump']:
                # 큰 변화가 발생하면 현재 검출을 무시하고 이전 정상 위치 사용
                self.ball_history.append(self.last_valid_ball_center)
                return False

        # 정상 검출로 처리
        self.ball_history.append(current_center)
        self.last_valid_ball_center = current_center
        self.last_valid_frame_idx = current_frame_idx
        self.valid_detections.append((current_frame_idx, current_center))

        # 이력이 충분하지 않으면 False 반환
        if len(self.ball_history) < self.shooting_threshold['trajectory_points']:
            return False

        recent_points = [p for p in self.ball_history[-self.shooting_threshold['detection_window']:] if p is not None]
        if len(recent_points) < self.shooting_threshold['trajectory_points']:
            return False

        # 수직 방향 분석
        vertical_movements = [recent_points[i][1] - recent_points[i-1][1] 
                              for i in range(1, len(recent_points))]
        # 수평 방향 분석
        horizontal_movements = [recent_points[i][0] - recent_points[i-1][0] 
                                for i in range(1, len(recent_points))]

        total_vertical_change = recent_points[-1][1] - recent_points[0][1]
        total_horizontal_change = recent_points[-1][0] - recent_points[0][0]

        vertical_speeds = [abs(v) for v in vertical_movements]
        avg_vertical_speed = np.mean(vertical_speeds)

        vertical_accelerations = [vertical_speeds[i] - vertical_speeds[i-1] 
                                  for i in range(1, len(vertical_speeds))]

        curvatures = []
        for i in range(1, len(recent_points)-1):
            p0, p1, p2 = recent_points[i-1], recent_points[i], recent_points[i+1]
            area = abs(0.5 * (p0[0]*(p1[1]-p2[1]) + p1[0]*(p2[1]-p0[1]) + p2[0]*(p0[1]-p1[1])))
            d01 = math.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)
            d12 = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            d02 = math.sqrt((p2[0]-p0[0])**2 + (p2[1]-p0[1])**2)
            if d01 * d12 * d02 != 0:
                curvature = (4 * area) / (d01 * d12 * d02)
                curvatures.append(curvature)
            else:
                curvatures.append(0)

        avg_curvature = np.mean(curvatures) if curvatures else 0

        self.debug_data = {
            'avg_vertical_speed': avg_vertical_speed,
            'total_vertical_change': total_vertical_change,
            'total_horizontal_change': total_horizontal_change,
            'avg_curvature': avg_curvature,
            'recent_points': recent_points,
            'vertical_speeds': vertical_speeds,
            'vertical_accelerations': vertical_accelerations
        }

        condition_velocity = avg_vertical_speed > self.shooting_threshold['vertical_velocity']
        condition_height = abs(total_vertical_change) > self.shooting_threshold['min_height_change']
        condition_acceleration = any(acc > self.shooting_threshold['vertical_velocity'] for acc in vertical_accelerations)
        condition_curvature = avg_curvature > self.shooting_threshold['min_curve']
        upward_movements_ratio = sum(1 for v in vertical_movements if v < 0) / len(vertical_movements) > 0.7

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

        # 유효한 디버그 데이터 업데이트
        self.last_debug_data = self.debug_data.copy()

        if len(self.ball_history) > self.shooting_threshold['detection_window']:
            self.ball_history.pop(0)

        return is_shooting

def visualize_shooting_data(frame, shooting_detector):
    """슈팅 관련 데이터를 프레임에 시각화"""
    # 사용 가능한 디버그 데이터 가져오기
    if shooting_detector.debug_data:
        data = shooting_detector.debug_data
    elif shooting_detector.last_debug_data:
        data = shooting_detector.last_debug_data
    else:
        data = None

    if data is not None:
        text_x = 50
        text_y = 50
        line_height = 30

        # 기본 데이터 표시
        cv2.putText(frame, f"Avg Speed: {data['avg_vertical_speed']:.2f}", 
                    (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"Vertical Change: {data['total_vertical_change']:.2f}", 
                    (text_x, text_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"Horizontal Change: {data['total_horizontal_change']:.2f}",
                    (text_x, text_y + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"Curvature: {data['avg_curvature']:.4f}", 
                    (text_x, text_y + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # 조건 상태 표시
        if 'conditions' in data:
            for i, (condition, is_met) in enumerate(data['conditions'].items()):
                color = (0,255,0) if is_met else (0,0,255)
                cv2.putText(frame, f'{condition}: {is_met}',
                           (text_x, text_y + line_height * (i+5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 궤적 그리기
        if 'recent_points' in data:
            points = data['recent_points']
            for i in range(1, len(points)):
                pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                pt2 = (int(points[i][0]), int(points[i][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    else:
        # 데이터가 없는 경우 "No Detection" 표시
        text_x = 50
        text_y = 50
        line_height = 30
        cv2.putText(frame, "No Detection", 
                    (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    return frame

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

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

def compute_average_speed(valid_detections, frame_window):
    """
    주어진 프레임 윈도우 내에서의 평균 속도를 계산.
    valid_detections: (frame_idx, center) 튜플 리스트
    frame_window: 평균 계산에 사용할 프레임 수
    """
    if len(valid_detections) < 2:
        return 0

    recent_valid = [d for d in valid_detections if d[0] >= valid_detections[-1][0] - frame_window]
    if len(recent_valid) < 2:
        return 0

    start_frame, start_center = recent_valid[0]
    end_frame, end_center = recent_valid[-1]
    distance = math.sqrt((end_center[0] - start_center[0])**2 + (end_center[1] - start_center[1])**2)
    frame_diff = end_frame - start_frame

    if frame_diff == 0:
        return 0

    avg_speed = distance / frame_diff
    return avg_speed

def main():
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

    cap = cv2.VideoCapture('dataset1/kbl_input2.mov')
    if not cap.isOpened():
        print("Error: Cannot open input video")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('dataset1/kbl_output2.mp4', fourcc, fps, (width, height))

    pbar = tqdm(total=total_frames, desc="Processing frames")

    shooting_detector = ShootingDetector()
    frame_idx = 0

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

        # Shooting 감지 및 시각화
        if ball_box is not None:
            is_shooting = shooting_detector.analyze_ball_trajectory(ball_box, frame_idx)
            # analyze_ball_trajectory가 True 또는 False를 반환하지만, debug_data는 유효한 경우에만 업데이트
            frame = visualize_shooting_data(frame, shooting_detector)

            # 이전 정상 위치와 현재 중심 사이 선 시각화 (debug_data가 있는 경우)
            if shooting_detector.debug_data:
                bx1, by1, bx2, by2 = ball_box
                current_center = ((bx1 + bx2) // 2, (by1 + by2) // 2)
                prev_center = shooting_detector.last_valid_ball_center
                cv2.line(frame, (int(prev_center[0]), int(prev_center[1])), 
                          (int(current_center[0]), int(current_center[1])), 
                          (255, 0, 255), 2)

            avg_speed = compute_average_speed(shooting_detector.valid_detections, 
                                            shooting_detector.shooting_threshold['detection_window'])
            cv2.putText(frame, f"Avg Speed Filtered: {avg_speed:.2f}", (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            status_text = "SHOOTING!" if is_shooting else "No shooting"
            status_color = (0,255,0) if is_shooting else (0,0,255)
            cv2.putText(frame, status_text, (50, height - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
        else:
            # 공이 감지되지 않은 경우에도 시각화 데이터 표시
            frame = visualize_shooting_data(frame, shooting_detector)

        # 사람 탐지 및 포즈 추정
        person_results = person_model.predict(frame, imgsz=640, conf=0.4, verbose=False)
        det = person_results[0].boxes
        boxes = det.xyxy.cpu().numpy() if torch.cuda.is_available() else det.xyxy.numpy()
        cls = det.cls.cpu().numpy() if torch.cuda.is_available() else det.cls.numpy()

        ball_center = None
        if ball_box is not None:
            bx1, by1, bx2, by2 = ball_box
            ball_center = ((bx1 + bx2) // 2, (by1 + by2) // 2)
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

                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    if (start_idx in landmarks_px) and (end_idx in landmarks_px):
                        cv2.line(frame,
                                landmarks_px[start_idx],
                                landmarks_px[end_idx],
                                (0, 255, 0), 2)

                for idx in landmarks_px:
                    cv2.circle(frame, landmarks_px[idx], 4, (0, 0, 255), -1)

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
        frame_idx += 1

    pbar.close()
    cap.release()
    out.release()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nProcessing complete!")
    print("Output video saved as 'dataset1/kbl_output.mp4'")

if __name__ == "__main__":
    main()
