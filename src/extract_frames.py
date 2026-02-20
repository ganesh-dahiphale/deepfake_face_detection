import cv2
import os

video_root = r"Deepfake_detection_project/data/raw_videos"
frame_root = "data/frames"

for label in os.listdir(video_root):
    label_path = os.path.join(video_root, label)

    for video in os.listdir(label_path):

        video_path = os.path.join(label_path, video)
        save_folder = os.path.join(frame_root, label, video[:-4])
        os.makedirs(save_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % 5 == 0:
                cv2.imwrite(f"{save_folder}/{count}.jpg", frame)

            count += 1

        cap.release()
print("Frame extraction completed.")