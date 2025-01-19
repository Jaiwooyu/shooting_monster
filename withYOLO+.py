import cv2
import torch
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """
    세 점 (a, b, c)의 2D 좌표 (x, y)를 받아,
    b를 꼭짓점으로 하는 각도(0 ~ 180도)를 반환.
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
    어깨 중점 -> 골반 중점 벡터가 수직(y축)과 이루는 간단한 각도.
    0도 근처 = 몸통 수직 / ±90도 근처 = 거의 수평
    """
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    angle_deg = np.degrees(np.arctan2(dy, dx))
    # 수직이 0도가 되도록 -90 보정
    trunk_angle = angle_deg - 90
    return trunk_angle

def main():
    # -----------------------------
    # 1) YOLOv5 모델 로드 (COCO 사전 학습)
    # -----------------------------
    #  - model='yolov5s'  : 작은 버전
    #  - 만약 custom 모델(best.pt)이 있다면 아래처럼:
    #    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    #  - GPU 사용하려면, device='cuda' 설정 등 추가
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = 0.4  # confidence threshold
    model.iou = 0.45  # IoU threshold

    # -----------------------------
    # 2) 비디오 열기 (입력/출력)
    # -----------------------------
    input_video_path = 'input_video.mp4'
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {input_video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        'output_basketball_analysis.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (width, height)
    )

    # -----------------------------
    # 3) MediaPipe Pose 초기화
    # -----------------------------
    pose_estimator = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frame_count = 0

    # -----------------------------
    # 4) 메인 루프
    # -----------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nNo more frames. Exiting...")
            break

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processing frame {frame_count}...", end='\r')

        # --------------------------------
        # (A) YOLOv5 추론 -> 농구공(sports ball, class=32) 검출
        # --------------------------------
        results = model(frame, size=640)
        df = results.pandas().xyxy[0]  # bbox 결과를 pandas DataFrame으로

        # 농구공 표시(바운딩박스)
        for i in range(len(df)):
            cls_id = df.loc[i, 'class']
            conf   = df.loc[i, 'confidence']
            if int(cls_id) == 32:  # sports ball
                x1 = int(df.loc[i, 'xmin'])
                y1 = int(df.loc[i, 'ymin'])
                x2 = int(df.loc[i, 'xmax'])
                y2 = int(df.loc[i, 'ymax'])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)
                cv2.putText(frame, f'Ball {conf:.2f}', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # --------------------------------
        # (B) MediaPipe Pose -> 스켈레톤 & 관절 각도
        # --------------------------------
        # BGR->RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose_estimator.process(frame_rgb)

        if results_pose.pose_landmarks:
            # (1) 스켈레톤 그리기
            mp_drawing.draw_landmarks(
                frame,
                results_pose.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
            )

            # (2) 관절 픽셀 좌표 추출
            hF, wF = frame.shape[:2]
            lm = results_pose.pose_landmarks.landmark

            # 랜드마크 인덱스 참조
            # https://google.github.io/mediapipe/solutions/pose.html
            r_shoulder = (int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * wF),
                          int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * hF))
            r_elbow    = (int(lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * wF),
                          int(lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * hF))
            r_wrist    = (int(lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * wF),
                          int(lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * hF))

            l_shoulder = (int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * wF),
                          int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * hF))
            l_elbow    = (int(lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * wF),
                          int(lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * hF))
            l_wrist    = (int(lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x * wF),
                          int(lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y * hF))

            r_hip      = (int(lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * wF),
                          int(lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * hF))
            r_knee     = (int(lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * wF),
                          int(lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * hF))
            r_ankle    = (int(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * wF),
                          int(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * hF))

            l_hip      = (int(lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * wF),
                          int(lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * hF))
            l_knee     = (int(lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x * wF),
                          int(lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y * hF))
            l_ankle    = (int(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * wF),
                          int(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * hF))

            # (3) 각도 계산
            angle_r_elbow = calculate_angle(r_shoulder, r_elbow, r_wrist)
            angle_l_elbow = calculate_angle(l_shoulder, l_elbow, l_wrist)
            angle_r_knee  = calculate_angle(r_hip, r_knee, r_ankle)
            angle_l_knee  = calculate_angle(l_hip, l_knee, l_ankle)

            # 허리(몸통) 각도
            shoulder_mid = ((r_shoulder[0]+l_shoulder[0])//2, (r_shoulder[1]+l_shoulder[1])//2)
            hip_mid      = ((r_hip[0]+l_hip[0])//2, (r_hip[1]+l_hip[1])//2)
            trunk_angle  = calculate_trunk_angle(shoulder_mid, hip_mid)

            # (4) 각도 텍스트 오버레이
            cv2.putText(frame, f'R_Elbow:{int(angle_r_elbow)}',
                        (r_elbow[0]+10, r_elbow[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f'L_Elbow:{int(angle_l_elbow)}',
                        (l_elbow[0]+10, l_elbow[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f'R_Knee:{int(angle_r_knee)}',
                        (r_knee[0]+10, r_knee[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f'L_Knee:{int(angle_l_knee)}',
                        (l_knee[0]+10, l_knee[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f'Trunk:{int(trunk_angle)}',
                        (shoulder_mid[0], shoulder_mid[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # --------------------------------
        # (C) 최종 프레임을 VideoWriter에 기록
        # --------------------------------
        out.write(frame)

    cap.release()
    out.release()
    print("\nProcessing complete!")
    print("Output file: 'output_basketball_analysis.mp4'")

if __name__ == "__main__":
    main()
