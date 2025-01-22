import bpy
import numpy as np
import math

def setup_blender_scene():
    """Blender 장면 초기화 및 설정"""
    # 기존 객체들 삭제
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # 조명 설정
    bpy.ops.object.light_add(type='SUN')
    sun = bpy.context.active_object
    sun.location = (5, 5, 10)
    sun.data.energy = 5
    
    # 카메라 설정
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.location = (0, -5, 2)
    camera.rotation_euler = (math.radians(75), 0, 0)
    bpy.context.scene.camera = camera

def create_joint_material():
    """관절을 위한 재질 생성"""
    mat = bpy.data.materials.new(name="JointMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes["Principled BSDF"].inputs[0].default_value = (1, 0, 0, 1)  # 빨간색
    return mat

def create_bone_material():
    """뼈대를 위한 재질 생성"""
    mat = bpy.data.materials.new(name="BoneMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes["Principled BSDF"].inputs[0].default_value = (0, 0, 1, 1)  # 파란색
    return mat

def create_skeleton_animation(features, fps=30):
    """
    포즈 시퀀스의 3D 스켈레톤 애니메이션을 Blender로 생성
    
    Args:
        features: NumPy 배열, shape (프레임 수, 132)
        fps: 초당 프레임 수
    """
    setup_blender_scene()
    
    # 재질 생성
    joint_material = create_joint_material()
    bone_material = create_bone_material()
    
    # 관절 연결 정의
    skeleton_connections = [
        (11, 13), (13, 15),  # 왼쪽 어깨 -> 팔꿈치 -> 손목
        (12, 14), (14, 16),  # 오른쪽 어깨 -> 팔꿈치 -> 손목
        (11, 12),  # 양 어깨 연결
        (23, 25), (25, 27),  # 왼쪽 엉덩이 -> 무릎 -> 발목
        (24, 26), (26, 28),  # 오른쪽 엉덩이 -> 무릎 -> 발목
        (11, 23), (12, 24)   # 어깨와 엉덩이 연결
    ]
    
    # 프레임 설정
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = len(features) - 1
    scene.render.fps = fps
    
    # 관절 객체 생성 및 애니메이션
    joints = []
    for i in range(33):  # 33개 관절
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.03)
        joint = bpy.context.active_object
        joint.name = f"Joint_{i}"
        joint.data.materials.append(joint_material)
        joints.append(joint)
    
    # 뼈대 객체 생성
    bones = []
    for connection in skeleton_connections:
        joint1, joint2 = connection
        
        # 실린더로 뼈대 생성
        bpy.ops.mesh.primitive_cylinder_add(radius=0.02)
        bone = bpy.context.active_object
        bone.name = f"Bone_{joint1}_{joint2}"
        bone.data.materials.append(bone_material)
        bones.append(bone)
    
    # 키프레임 애니메이션 생성
    for frame_idx, frame_data in enumerate(features):
        scene.frame_set(frame_idx)
        
        # 관절 위치 업데이트
        joints_data = frame_data.reshape(-1, 4)[:, :3]  # (33, 3) 형태로 변환
        
        for i, joint in enumerate(joints):
            joint.location = joints_data[i]
            joint.keyframe_insert(data_path="location")
        
        # 뼈대 위치 및 방향 업데이트
        for bone, connection in zip(bones, skeleton_connections):
            joint1, joint2 = connection
            start_pos = joints_data[joint1]
            end_pos = joints_data[joint2]
            
            # 뼈대 위치 및 방향 계산
            center = (start_pos + end_pos) / 2
            direction = end_pos - start_pos
            length = np.linalg.norm(direction)
            
            if length > 0:
                bone.scale = (1, 1, length)
                bone.location = center
                
                # 방향 계산
                rot_quat = direction_to_quaternion(direction)
                bone.rotation_mode = 'QUATERNION'
                bone.rotation_quaternion = rot_quat
                
                # 키프레임 삽입
                bone.keyframe_insert(data_path="location")
                bone.keyframe_insert(data_path="rotation_quaternion")
                bone.keyframe_insert(data_path="scale")

def direction_to_quaternion(direction):
    """방향 벡터를 쿼터니언으로 변환"""
    direction = direction / np.linalg.norm(direction)
    
    # Z축을 기준으로 회전
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, direction)
    
    if np.allclose(rotation_axis, 0):
        if direction[2] > 0:
            return (1, 0, 0, 0)
        else:
            return (0, 1, 0, 0)
    
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    angle = np.arccos(np.dot(z_axis, direction))
    
    return angle_axis_to_quaternion(angle, rotation_axis)

def angle_axis_to_quaternion(angle, axis):
    """각도와 축을 쿼터니언으로 변환"""
    s = math.sin(angle/2)
    return (math.cos(angle/2), axis[0]*s, axis[1]*s, axis[2]*s)

def generate_test_motion(num_frames=60):
    """
    테스트용 동작 데이터를 생성합니다.
    간단한 슈팅 동작을 시뮬레이션합니다.
    
    Args:
        num_frames: 생성할 프레임 수
        
    Returns:
        NumPy 배열, shape (num_frames, 132)
    """
    features = []
    for i in range(num_frames):
        # 기본 자세 생성
        frame = np.zeros((33, 4))  # 33개 관절, (x, y, z, visibility)
        
        # 시간에 따른 위치 변화 (0~1 범위)
        t = i / num_frames
        
        # 슈팅 동작 시뮬레이션
        # 몸통 관절 (11, 12, 23, 24)
        frame[[11, 12], 0] = 0  # 어깨 x 좌표
        frame[[11, 12], 1] = 1 + np.sin(t * np.pi) * 0.2  # 어깨 y 좌표
        frame[[11, 12], 2] = np.sin(t * np.pi) * 0.1  # 어깨 z 좌표
        
        frame[[23, 24], 0] = 0  # 엉덩이 x 좌표
        frame[[23, 24], 1] = 0.5 + np.sin(t * np.pi) * 0.3  # 엉덩이 y 좌표
        
        # 오른팔 관절 (14, 16) - 슈팅 팔
        frame[14, 0] = 0.3  # 팔꿈치 x
        frame[14, 1] = 0.8 + t * 0.5  # 팔꿈치 y
        frame[14, 2] = 0.2 + t * 0.8  # 팔꿈치 z
        
        frame[16, 0] = 0.4  # 손목 x
        frame[16, 1] = 1.0 + t * 0.8  # 손목 y
        frame[16, 2] = 0.4 + t * 1.2  # 손목 z
        
        # 왼팔 관절 (13, 15) - 보조 팔
        frame[13, 0] = -0.3
        frame[13, 1] = 0.8 + t * 0.3
        frame[13, 2] = 0.2 + t * 0.4
        
        frame[15, 0] = -0.4
        frame[15, 1] = 1.0 + t * 0.4
        frame[15, 2] = 0.3 + t * 0.6
        
        # 다리 관절 (25, 26, 27, 28)
        frame[[25, 26], 1] = -0.5 + np.sin(t * np.pi) * 0.2  # 무릎
        frame[[27, 28], 1] = -1.0  # 발목
        
        # visibility 설정
        frame[:, 3] = 1.0  # 모든 관절이 보임
        
        features.append(frame.flatten())
    
    return np.array(features)

# 사용 예시:
if __name__ == "__main__":
    # 테스트 데이터 생성 또는 실제 데이터 로드
    test_features = generate_test_motion(num_frames=60)  # 이전 코드의 generate_test_motion 함수 사용
    
    # Blender 애니메이션 생성
    create_skeleton_animation(test_features, fps=30)
    
    # 렌더링 설정
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080