import numpy as np
import cv2
import tensorflow as tf

def calculate_angle(point1, point2, point3):
    """
    세 점 사이의 각도를 계산합니다.
    
    Args:
        point1, point2, point3: numpy array 형태의 좌표점
        
    Returns:
        float: 각도 (도 단위)
    """
    ba = point1 - point2
    bc = point3 - point2
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def check_elbow_quality(keypoints):
    """
    팔꿈치 각도의 품질을 체크합니다.
    
    Args:
        keypoints: OpenPose에서 추출한 키포인트 배열
        
    Returns:
        float: 품질 점수 (0.4 ~ 1.0)
    """
    shoulder = keypoints[2]
    elbow = keypoints[3]
    wrist = keypoints[4]
    
    angle = calculate_angle(shoulder, elbow, wrist)
    
    if 85 <= angle <= 95:
        return 1.0
    elif 80 <= angle <= 100:
        return 0.8
    elif 75 <= angle <= 105:
        return 0.6
    else:
        return 0.4

def check_knee_quality(keypoints):
    """
    무릎 굽힘의 품질을 체크합니다.
    
    Args:
        keypoints: OpenPose에서 추출한 키포인트 배열
        
    Returns:
        float: 품질 점수 (0.4 ~ 1.0)
    """
    hip = keypoints[9]
    knee = keypoints[10]
    ankle = keypoints[11]
    
    angle = calculate_angle(hip, knee, ankle)
    
    if 130 <= angle <= 150:
        return 1.0
    elif 120 <= angle <= 160:
        return 0.8
    elif 110 <= angle <= 170:
        return 0.6
    else:
        return 0.4

def check_balance_quality(keypoints):
    """
    전체적인 자세 밸런스를 체크합니다.
    
    Args:
        keypoints: OpenPose에서 추출한 키포인트 배열
        
    Returns:
        float: 품질 점수 (0.4 ~ 1.0)
    """
    ankle = keypoints[11]
    hip = keypoints[9]
    
    vertical_alignment = abs(ankle[0] - hip[0])
    
    if vertical_alignment < 20:
        return 1.0
    elif vertical_alignment < 30:
        return 0.8
    elif vertical_alignment < 40:
        return 0.6
    else:
        return 0.4

def detect_ball(frame, detection_graph, sess):
    """
    TensorFlow 모델을 사용하여 농구공을 감지합니다.
    
    Args:
        frame: 입력 이미지 프레임
        detection_graph: TensorFlow 감지 그래프
        sess: TensorFlow 세션
        
    Returns:
        bool: 농구공 감지 여부
    """
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    
    frame_expanded = np.expand_dims(frame, axis=0)
    (boxes, scores, classes) = sess.run(
        [detection_boxes, detection_scores, detection_classes],
        feed_dict={image_tensor: frame_expanded}
    )
    
    return np.any(scores > 0.5) and np.any(classes == 1)  # 클래스 1이 농구공

def check_shot_form_quality(keypoints):
    """
    전체적인 슛 폼의 품질을 평가합니다.
    
    Args:
        keypoints: OpenPose에서 추출한 키포인트 배열
        
    Returns:
        float: 종합 품질 점수 (0.4 ~ 1.0)
    """
    elbow_score = check_elbow_quality(keypoints)
    knee_score = check_knee_quality(keypoints)
    balance_score = check_balance_quality(keypoints)
    
    weighted_score = (
        elbow_score * 0.4 +    # 팔꿈치 각도 중요도 40%
        knee_score * 0.3 +     # 무릎 굽힘 중요도 30%
        balance_score * 0.3    # 전체 밸런스 중요도 30%
    )
    
    return weighted_score

def analyze_shooting_sequence(sequence_keypoints):
    """
    전체 슛 동작 시퀀스를 분석합니다.
    
    Args:
        sequence_keypoints: 시퀀스의 키포인트 배열 목록
        
    Returns:
        dict: 분석 결과를 담은 딕셔너리
    """
    sequence_scores = []
    for keypoints in sequence_keypoints:
        frame_score = check_shot_form_quality(keypoints)
        sequence_scores.append(frame_score)
    
    sequence_scores = np.array(sequence_scores)
    
    return {
        'average_score': np.mean(sequence_scores),
        'max_score': np.max(sequence_scores),
        'min_score': np.min(sequence_scores),
        'consistency': np.std(sequence_scores)
    }