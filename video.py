import cv2
import mediapipe as mp
import numpy as np
import collections

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """
    세 점 (a, b, c)의 좌표를 기반으로 b점을 꼭짓점으로 하는 각도를 계산하는 함수.
    a, b, c는 [x, y] 형태 (픽셀 좌표)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def main():
    cap = cv2.VideoCapture('input_video.mp4')
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # 결과 저장용 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_basketball_analysis.mp4', fourcc, fps, (width, height))

    # 공의 중심점 궤적을 저장할 큐(지나간 점들을 저장하여 라인 그리기)
    # collections.deque를 사용하면 일정 길이 이상의 데이터가 자동으로 정리됨
    traj_length = 30
    ball_trajectory = collections.deque(maxlen=traj_length)

    # 골대 영역(예시): 단순히 (xmin, ymin), (xmax, ymax) 형태로 지정
    # 실제론 객체 검출하여 자동으로 인식하거나, 더 정교한 방법을 써야 함
    hoop_xmin, hoop_ymin = int(width * 0.8), int(height * 0.2)
    hoop_xmax, hoop_ymax = int(width * 0.95), int(height * 0.4)

    # MediaPipe Pose 초기화
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        shot_made = False  # 슛 성공 여부 플래그

        while True:
            success, frame = cap.read()
            if not success:
                break

            # BGR -> RGB (MediaPipe용)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # ------------------------------------------
            # (A) 농구공 검출 (색 기반 간단 예시)
            # ------------------------------------------
            # 1) BGR -> HSV 변환
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 2) 농구공 색 범위 설정 (예: 주황~갈색) -> 실험적으로 조정 필요
            #    [H, S, V] -> 0~179, 0~255, 0~255 (OpenCV 기준)
            lower_orange = np.array([5, 100, 100])
            upper_orange = np.array([20, 255, 255])
            mask = cv2.inRange(hsv, lower_orange, upper_orange)

            # 3) 모폴로지 연산(노이즈 제거)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # 4) 윤곽선 검출
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ball_center = None

            if contours:
                # 가장 큰 컨투어를 농구공으로 가정
                c = max(contours, key=cv2.contourArea)
                # 공의 외접원(중심, 반지름) 구하기
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                
                if radius > 5:  # 임계값 (너무 작은 객체 제외)
                    ball_center = (int(x), int(y))
                    # 원 그리기
                    cv2.circle(frame, ball_center, int(radius), (0, 140, 255), 2)

            # ball_center가 유효하면 궤적에 저장
            if ball_center is not None:
                ball_trajectory.append(ball_center)
            
            # 공 궤적(라인) 시각화
            for i in range(1, len(ball_trajectory)):
                cv2.line(frame,
                         ball_trajectory[i - 1],
                         ball_trajectory[i],
                         (0, 255, 255), 3)

            # ------------------------------------------
            # (B) 골대(hoop) 표시 & 슛 성공 여부 체크
            # ------------------------------------------
            cv2.rectangle(frame, (hoop_xmin, hoop_ymin), (hoop_xmax, hoop_ymax), (0, 0, 255), 2)
            # 단순판정: 공 중심이 골대 영역을 통과하면 "슛 성공" 처리
            if not shot_made and ball_center is not None:
                bx, by = ball_center
                if hoop_xmin < bx < hoop_xmax and hoop_ymin < by < hoop_ymax:
                    shot_made = True  # 한 번 통과하면 성공

            # 성공 여부 표시
            if shot_made:
                cv2.putText(frame, "SHOT MADE!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

            # ------------------------------------------
            # (C) MediaPipe Pose로 선수 자세(각도 등) 분석
            # ------------------------------------------
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # 픽셀 좌표 변환용 함수
                def to_pixel(landmark):
                    return (int(landmark.x * width), int(landmark.y * height))
                
                # 예: 오른쪽 어깨 / 팔꿈치 / 손목 각도
                r_shoulder = to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
                r_elbow    = to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
                r_wrist    = to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
                angle_r_elbow = calculate_angle(r_shoulder, r_elbow, r_wrist)

                cv2.putText(frame, f'R_Elbow: {int(angle_r_elbow)} deg', 
                            r_elbow, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                # 예: 왼쪽 팔꿈치 각도
                l_shoulder = to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
                l_elbow    = to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
                l_wrist    = to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
                angle_l_elbow = calculate_angle(l_shoulder, l_elbow, l_wrist)
                cv2.putText(frame, f'L_Elbow: {int(angle_l_elbow)} deg', 
                            l_elbow, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                # 예: 오른쪽 골반 / 무릎 / 발목 각도
                r_hip   = to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
                r_knee  = to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
                r_ankle = to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
                angle_r_knee = calculate_angle(r_hip, r_knee, r_ankle)

                cv2.putText(frame, f'R_Knee: {int(angle_r_knee)} deg', 
                            r_knee, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                # 마찬가지로 왼쪽 무릎, 손목 각도 등 추가 가능

                # 관절 랜드마크 시각화
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

            # ------------------------------------------
            # (D) 경기장(코트) 분석 (간단 예시)
            # ------------------------------------------
            # 예: 코트 영역을 임의로 지정해, 그 영역 내에서만 데이터 유효 처리
            # 실제론 라인 검출(허프 변환 등)이나 세그멘테이션이 필요할 수 있음
            # 여기서는 시각화만 간단히 표시
            cv2.rectangle(frame, (int(width*0.05), int(height*0.1)),
                                 (int(width*0.75), int(height*0.9)), (255, 0, 0), 2)
            cv2.putText(frame, "Court Region (example)", (int(width*0.05), int(height*0.09)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            # ------------------------------------------
            # (E) 최종 결과 저장
            # ------------------------------------------
            out.write(frame)

        cap.release()
        out.release()

if __name__ == "__main__":
    main()
