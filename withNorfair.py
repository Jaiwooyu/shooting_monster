import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker
from tqdm import tqdm

# Norfair 거리 함수 정의: 두 점 사이의 유클리드 거리
def distance_function(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

# YOLO 모델 초기화 (농구공 검출)
ball_model = YOLO('best.pt')

# Norfair 트래커 초기화
tracker = Tracker(distance_function=distance_function, distance_threshold=50)

# 비디오 캡처 및 출력 설정
cap = cv2.VideoCapture('dataset1/input_video.mp4')
if not cap.isOpened():
    print("Error: Cannot open input_video.mp4")
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_norfair_basketball.mp4', fourcc, fps, (width, height))

pbar = tqdm(total=total_frames, desc="Processing frames")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO를 이용한 농구공 검출 (이미지 크기를 32 배수로 조정하여 경고 회피)
    results = ball_model.predict(frame, imgsz=[1920, 1088], conf=0.1, verbose=False)
    ball_boxes = results[0].boxes.xyxy.cpu().numpy()

    detections = []
    # 각 검출된 박스에 대해 중심 좌표를 계산하여 Norfair Detection 객체 생성
    for box in ball_boxes:
        x1, y1, x2, y2 = box
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        detections.append(Detection(points=center))

    # Norfair 트래커 업데이트
    tracked_objects = tracker.update(detections=detections)

    # 추적 결과 시각화
    for obj in tracked_objects:
        # obj.estimate를 평탄화하여 중심 좌표 추출
        estimate = obj.estimate.flatten()
        center_x, center_y = int(estimate[0]), int(estimate[1])
        cv2.circle(frame, (center_x, center_y), 15, (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {obj.id}', (center_x, center_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 결과 프레임 저장
    out.write(frame)

    pbar.update(1)

pbar.close()
cap.release()
out.release()

print("Processing complete! Output saved as 'output_norfair_basketball.mp4'.")
