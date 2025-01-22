import cv2
import mediapipe as mp
import plotly.graph_objects as go

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Video 파일 열기
cap = cv2.VideoCapture("dataset1/input_video.mp4")

# 결과 저장 경로
output_image_path = "output_frame.jpg"
output_plot_path = "output_plot.html"

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR 이미지를 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Pose 추정
        results = pose.process(image)

        # 랜드마크 데이터 추출 및 Plotly 3D 시각화
        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            x = [landmark.x for landmark in landmarks]
            y = [landmark.y for landmark in landmarks]
            z = [landmark.z for landmark in landmarks]

            # Plotly 3D 플롯 생성
            fig = go.Figure(data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=5,
                    color=z,  # Z 좌표에 따라 색상 변경
                    colorscale='Viridis',
                    opacity=0.8
                )
            )])

            # 레이아웃 설정
            fig.update_layout(
                scene=dict(
                    xaxis=dict(nticks=10, range=[-1, 1]),
                    yaxis=dict(nticks=10, range=[-1, 1]),
                    zaxis=dict(nticks=10, range=[-1, 1]),
                ),
                margin=dict(r=0, l=0, b=0, t=0)  # 여백 최소화
            )

            # 3D 플롯 HTML로 저장
            fig.write_html(output_plot_path)
            print(f"Saved 3D plot to {output_plot_path}")

        # 랜드마크 시각화된 프레임 저장
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imwrite(output_image_path, image)
        print(f"Saved frame to {output_image_path}")

        # 한 프레임만 저장 후 종료
        break

cap.release()
