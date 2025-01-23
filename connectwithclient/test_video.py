import cv2
import sys

def test_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}", file=sys.stderr)
        return
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
    cap.release()
    print(f"Successfully read {frame_count} frames from video.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_video.py <video_path>", file=sys.stderr)
        sys.exit(1)
    video_path = sys.argv[1]
    test_video(video_path)
