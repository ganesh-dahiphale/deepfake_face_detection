from retinaface import RetinaFace
import cv2, os

frame_root = "data/frames"
face_root = "data/faces"

for label in os.listdir(frame_root):

    for video in os.listdir(os.path.join(frame_root,label)):

        frame_dir = os.path.join(frame_root,label,video)

        for img in os.listdir(frame_dir):

            path = os.path.join(frame_dir,img)
            image = cv2.imread(path)

            faces = RetinaFace.detect_faces(path)

            if isinstance(faces, dict):

                for k in faces.keys():

                    x1,y1,x2,y2 = faces[k]["facial_area"]

                    crop = image[y1:y2, x1:x2]
                    crop = cv2.resize(crop,(224,224))

                    save_dir = os.path.join(face_root,label)
                    os.makedirs(save_dir,exist_ok=True)

                    cv2.imwrite(f"{save_dir}/{video}_{img}",crop)
print("Face extraction completed.")