from retinaface import RetinaFace
import cv2, os
from multiprocessing import Pool, cpu_count

frame_root="data/frames"
face_root="data/faces"

tasks=[]

for label in os.listdir(frame_root):

    for video in os.listdir(os.path.join(frame_root,label)):

        frame_dir=os.path.join(frame_root,label,video)

        for img in os.listdir(frame_dir):

            path=os.path.join(frame_dir,img)
            tasks.append((path,label,video,img))


def process(task):

    path,label,video,img=task

    image=cv2.imread(path)
    if image is None:
        return

    faces=RetinaFace.detect_faces(path)

    if isinstance(faces,dict):

        for k in faces:

            x1,y1,x2,y2=faces[k]["facial_area"]

            x1=max(0,x1); y1=max(0,y1)

            crop=image[y1:y2,x1:x2]
            if crop.size==0:
                return

            crop=cv2.resize(crop,(224,224))

            save_dir=os.path.join(face_root,label)
            os.makedirs(save_dir,exist_ok=True)

            cv2.imwrite(f"{save_dir}/{video}_{img}",crop)


if __name__=="__main__":

    print("Using CPUs:",cpu_count())

    with Pool(cpu_count()) as p:
        p.map(process,tasks)

    print("Multiprocessing extraction completed.")
