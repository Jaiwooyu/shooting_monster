import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO  # YOLOv8

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """
    세 점 (a, b, c)의 2D 좌표를 받아 b를 꼭짓점으로 하는 각도(0~180도)를 반환
    a, b, c: (x, y) 형태 튜플
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
    """
    어깨 중점(shoulder_mid) → 골반 중점(hip_mid) 벡터가 수직(y축)과 이루는 간단한 각도
    (농구 분석에서 '허리 각도'로 참고 가능)
    """
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    angle_deg = np.degrees(np.arctan2(dy, dx))  
    trunk_angle = angle_deg - 90  # 수직이 0도 되도록 보정
    return trunk_angle

def main():
    # 1) YOLO 모델 로드 (COCO 사전 학습)
    model = YOLO('yolov8n.pt')  # 혹은 yolov8s.pt 등

    # 2) MediaPipe Pose 초기화
    pose_estimator = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 3) 입력 영상 열기
    cap = cv2.VideoCapture('input_video.mp4')
    if not cap.isOpened():
        print("Error: Cannot open input_video.mp4")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # 4) 출력 동영상 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_skeleton_and_angles.mp4', fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nNo more frames. Exiting...")
            break

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processing frame {frame_count}...", end='\r')

        # 5) YOLO 추론
        results = model.predict(frame, imgsz=[width, height], conf=0.4, verbose=False)
        det = results[0].boxes

        boxes = det.xyxy.cpu().numpy()  # (N, 4)
        cls   = det.cls.cpu().numpy()   # (N,)
        confs = det.conf.cpu().numpy()  # (N,)

        person_box = None
        max_area   = 0

        # COCO: person=0
        for i in range(len(boxes)):
            class_id = int(cls[i])
            x1, y1, x2, y2 = boxes[i]
            w = x2 - x1
            h = y2 - y1
            area = w * h
            if class_id == 0:  # person
                if area > max_area:
                    max_area = area
                    person_box = (int(x1), int(y1), int(x2), int(y2))

        # (선택) 다른 객체(예: 농구공)도 검출 가능. 여기선 인체 분석에 집중.

        # 6) MediaPipe Pose
        if person_box is not None:
            px1, py1, px2, py2 = person_box
            px1 = max(0, px1); py1 = max(0, py1)
            px2 = min(width, px2); py2 = min(height, py2)

            # 해당 ROI를 추출
            person_roi = frame[py1:py2, px1:px2].copy()
            roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            pose_results = pose_estimator.process(roi_rgb)

            if pose_results.pose_landmarks:
                # (A) ROI상의 랜드마크를 원본 좌표로 매핑
                landmarks_px = {}
                for idx, lm in enumerate(pose_results.pose_landmarks.landmark):
                    cx = int(lm.x * (px2 - px1)) + px1
                    cy = int(lm.y * (py2 - py1)) + py1
                    landmarks_px[idx] = (cx, cy)

                # (B) 스켈레톤(신체 연결선) 직접 그리기
                # Mediapipe Pose 연결관계: mp_pose.POSE_CONNECTIONS
                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    if (start_idx in landmarks_px) and (end_idx in landmarks_px):
                        cv2.line(frame,
                                 landmarks_px[start_idx],
                                 landmarks_px[end_idx],
                                 (0, 255, 0), 2)  # 녹색 선

                # (C) 랜드마크 점
                for idx in landmarks_px:
                    cv2.circle(frame, landmarks_px[idx], 4, (0, 0, 255), -1)  # 빨간 점

                # (D) 주요 각도 계산
                # 예) 오른팔꿈치
                r_shoulder = landmarks_px[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                r_elbow    = landmarks_px[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                r_wrist    = landmarks_px[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                angle_r_elbow = calculate_angle(r_shoulder, r_elbow, r_wrist)

                # 왼팔꿈치
                l_shoulder = landmarks_px[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                l_elbow    = landmarks_px[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                l_wrist    = landmarks_px[mp_pose.PoseLandmark.LEFT_WRIST.value]
                angle_l_elbow = calculate_angle(l_shoulder, l_elbow, l_wrist)

                # 오른무릎
                r_hip   = landmarks_px[mp_pose.PoseLandmark.RIGHT_HIP.value]
                r_knee  = landmarks_px[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                r_ankle = landmarks_px[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                angle_r_knee = calculate_angle(r_hip, r_knee, r_ankle)

                # 왼무릎
                l_hip   = landmarks_px[mp_pose.PoseLandmark.LEFT_HIP.value]
                l_knee  = landmarks_px[mp_pose.PoseLandmark.LEFT_KNEE.value]
                l_ankle = landmarks_px[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                angle_l_knee = calculate_angle(l_hip, l_knee, l_ankle)

                # 허리(몸통) 각도
                shoulder_mid = ((r_shoulder[0] + l_shoulder[0]) // 2,
                                (r_shoulder[1] + l_shoulder[1]) // 2)
                hip_mid = ((r_hip[0] + l_hip[0]) // 2,
                           (r_hip[1] + l_hip[1]) // 2)
                trunk_angle = calculate_trunk_angle(shoulder_mid, hip_mid)

                # (E) 각도 정보 표시
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

            # 사람 박스도 표시
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
            cv2.putText(frame, "Player", (px1, py1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        # 7) 결과 프레임 저장
        out.write(frame)

    cap.release()
    out.release()

    print("\nProcessing complete!")
    print("Output video saved as 'output_skeleton_and_angles.mp4'")

if __name__ == "__main__":
    main()
