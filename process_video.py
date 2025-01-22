import os
import subprocess
import argparse
import sys
import time
import shutil
import tempfile

def process_video_to_fbx(input_video, output_dir, gender, person_id, fps_source, fps_target):
    try:
        input_video_name = os.path.basename(input_video)
        video_folder_name = input_video_name

        # 임시 디렉토리 생성 및 입력 비디오 복사
        temp_input_dir = tempfile.mkdtemp(prefix="temp_input_")
        temp_input_video = os.path.join(temp_input_dir, input_video_name)
        shutil.copy2(input_video, temp_input_video)

        output_path_abs = os.path.abspath(output_dir)

        docker_run_cmd = [
            'docker', 'run', '--gpus', 'all',
            '--rm',
            # 임시 입력 디렉토리를 읽기 전용으로 마운트
            '-v', f'{temp_input_dir}:/VIBE/input_videos:ro',
            '-v', f'{output_path_abs}:/VIBE/output_fbx',
            '--name', 'vibe_container',
            'vibe_env'
        ]

        commands = [
            'cd /VIBE',
            f'rm -rf /VIBE/output_fbx/{video_folder_name}',
            f'python3 demo.py --vid_file input_videos/{input_video_name} --output_folder output_fbx --save_obj',
            f'''blender -b --enable-autoexec -P <(echo 'import sys; sys.argv = ["blender"] + [
                "--input", "/VIBE/output_fbx/{video_folder_name}/vibe_output.pkl",
                "--output", "/VIBE/output_fbx/{video_folder_name}/{os.path.splitext(input_video_name)[0]}.fbx",
                "--fps_source", "{fps_source}",
                "--fps_target", "{fps_target}",
                "--gender", "{gender}",
                "--person_id", "{person_id}"
            ]') -P lib/utils/fbx_output.py'''
        ]

        full_command = docker_run_cmd + ['bash', '-c', ' && '.join(commands)]

        print("Starting VIBE processing pipeline...")
        print(f"Processing video: {input_video}")
        print(f"Output directory: {output_dir}")
        print("\nCleaning up existing files...")

        host_output_path = os.path.join(output_dir, 'output_fbx', video_folder_name)
        if os.path.exists(host_output_path):
            print(f"Removing existing output directory: {host_output_path}")
            shutil.rmtree(host_output_path, ignore_errors=True)

        process = subprocess.run(full_command,
                                 check=True,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 text=True)

        print("\nProcess output:")
        print(process.stdout)

        if process.stderr:
            print("\nProcess errors:")
            print(process.stderr)

        output_fbx = os.path.join(output_dir,
                                  'output_fbx',
                                  video_folder_name,
                                  f"{os.path.splitext(input_video_name)[0]}.fbx")

        if os.path.exists(output_fbx):
            print(f"\nSuccess! FBX file generated at: {output_fbx}")
            success = True
        else:
            print("\nError: FBX file was not generated")
            success = False

    except subprocess.CalledProcessError as e:
        print(f"\nError during processing: {str(e)}")
        print("\nError output:")
        print(e.stderr)
        success = False
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        success = False
    finally:
        # 임시 입력 디렉토리 및 파일 정리
        if os.path.exists(temp_input_dir):
            shutil.rmtree(temp_input_dir)

    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert video to FBX using VIBE')
    parser.add_argument('--input', required=True, help='Input video file path (.mp4, .mov, etc)')
    parser.add_argument('--output_dir', required=True, help='Output directory for FBX')
    parser.add_argument('--gender', default='male', choices=['male', 'female'], help='Gender for the model')
    parser.add_argument('--person_id', default=0, type=int, help='Person ID to track (default: 1)')
    parser.add_argument('--fps_source', default=30, type=int, help='Source video FPS (default: 30)')
    parser.add_argument('--fps_target', default=30, type=int, help='Target FBX FPS (default: 30)')

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Input video file not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()

    success = process_video_to_fbx(args.input,
                                   args.output_dir,
                                   args.gender,
                                   args.person_id,
                                   args.fps_source,
                                   args.fps_target)

    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")

    if not success:
        sys.exit(1)
