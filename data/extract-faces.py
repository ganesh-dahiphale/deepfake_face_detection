from retinaface import RetinaFace
import cv2
import os
from tqdm import tqdm
import numpy as np

frame_root = "data/frames"
face_root = "data/faces"

# ---------- Blur detection function ----------
def is_blurry(image, threshold=80):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap < threshold


# ---------- Collect all frame paths first (faster tqdm handling) ----------
all_images = []

for label in os.listdir(frame_root):
    label_path = os.path.join(frame_root, label)

    if not os.path.isdir(label_path):
        continue

    for video in os.listdir(label_path):
        frame_dir = os.path.join(label_path, video)

        if not os.path.isdir(frame_dir):
            continue

        for img in os.listdir(frame_dir):
            all_images.append((label, video, os.path.join(frame_dir, img)))


# ---------- Process with progress bar ----------
for label, video, path in tqdm(all_images, desc="Extracting Faces"):

    try:
        image = cv2.imread(path)

        # Skip unreadable images
        if image is None:
            continue

        # Detect faces (use image array -> faster than passing path)
        faces = RetinaFace.detect_faces(image)

        # Skip if no face detected
        if not isinstance(faces, dict):
            continue

        # ---------- Select ONLY the largest face ----------
        largest_face = None
        largest_area = 0

        for k in faces.keys():
            x1, y1, x2, y2 = faces[k]["facial_area"]

            # clamp coords (avoid negative / overflow)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            area = (x2 - x1) * (y2 - y1)

            if area > largest_area:
                largest_area = area
                largest_face = (x1, y1, x2, y2)

        if largest_face is None:
            continue

        x1, y1, x2, y2 = largest_face
        crop = image[y1:y2, x1:x2]

        # Skip tiny crops
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            continue

        # ---------- Skip blurry faces ----------
        if is_blurry(crop):
            continue

        # Resize for model input
        crop = cv2.resize(crop, (224, 224))

        # Save folder
        save_dir = os.path.join(face_root, label)
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{video}_{os.path.basename(path)}"
        cv2.imwrite(os.path.join(save_dir, filename), crop)

    except Exception as e:
        # silently skip problematic files
        continue
