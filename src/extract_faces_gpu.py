from retinaface import RetinaFace
import tensorflow as tf
import cv2, os

print("GPU devices:", tf.config.list_physical_devices('GPU'))

frame_root="data/frames"
face_root="data/faces"

for label in os.listdir(frame_root):

    label_path=os.path.join(frame_root,label)

    for video in os.listdir(label_path):

        frame_dir=os.path.join(label_path,video)

        for img in os.listdir(frame_dir):

            path=os.path.join(frame_dir,img)

            image=cv2.imread(path)
            if image is None:
                continue

            faces=RetinaFace.detect_faces(path)

            if isinstance(faces,dict):

                for k in faces:

                    x1,y1,x2,y2=faces[k]["facial_area"]

                    x1=max(0,x1); y1=max(0,y1)

                    crop=image[y1:y2,x1:x2]
                    if crop.size==0:
                        continue

                    crop=cv2.resize(crop,(224,224))

                    save_dir=os.path.join(face_root,label)
                    os.makedirs(save_dir,exist_ok=True)

                    cv2.imwrite(f"{save_dir}/{video}_{img}",crop)

print("GPU face extraction completed.")
