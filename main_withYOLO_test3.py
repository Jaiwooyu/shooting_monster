import cv2
import numpy as np
from ultralytics import YOLO

def visualize_pose(video_path, output_path):
    # YOLO 모델 로드
    pose_model = YOLO('yolov8l-pose.pt')
    
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    
    # 비디오 속성 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 결과 비디오 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 관절 연결 정의 (YOLOv8 pose estimation의 키포인트 순서 기준)
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], 
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], 
                [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # 진행 상황 출력
        frame_count += 1
        print(f'\rProcessing frame {frame_count}/{total_frames}', end='')
            
        # 포즈 추정 실행
        results = pose_model(frame)
        
        # 결과가 있는 경우에만 처리
        if len(results) > 0:
            # 각 감지된 사람에 대해 처리
            for person in results[0].keypoints.data:
                # 키포인트 그리기
                for kp in person:
                    x, y, conf = int(kp[0]), int(kp[1]), kp[2]
                    if conf > 0.5:  # 신뢰도가 0.5 이상인 키포인트만 표시
                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                
                # 스켈레톤 그리기
                for sk in skeleton:
                    pos1 = (int(person[sk[0]-1][0]), int(person[sk[0]-1][1]))
                    pos2 = (int(person[sk[1]-1][0]), int(person[sk[1]-1][1]))
                    
                    # 두 키포인트의 신뢰도가 모두 0.5 이상일 때만 선 그리기
                    if person[sk[0]-1][2] > 0.5 and person[sk[1]-1][2] > 0.5:
                        cv2.line(frame, pos1, pos2, (0, 0, 255), 2)
        
        # 결과 프레임 저장
        out.write(frame)
    
    print('\nProcessing completed!')
    
    # 리소스 해제
    cap.release()
    out.release()

# 사용 예시
video_path = 'input_video.mp4'  # 입력 비디오 경로
output_path = 'output_video.mp4'  # 출력 비디오 경로
visualize_pose(video_path, output_path)