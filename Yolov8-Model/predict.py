import os

from ultralytics import YOLO
import cv2


VIDEOS_DIR = os.path.join('.', 'videos')
IMG_DIR = os.path.join('.', 'img')

video_path = os.path.join(VIDEOS_DIR, 'Y2meta.mp4')

img_path = os.path.join(IMG_DIR, 'pigeon on the road_13.jpeg')
#video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
#out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'Model', 'bestV7.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.75

while ret:

    
    #results = model(source=video_path, stream=True,conf =threshold )  # generator of Results objects
    results = model(frame)[0]
    
    # for r in results:
    #     boxes = r.boxes  # Boxes object for bbox outputs
    #     key = r.keypoints
     
    #     print(f"{boxes}")

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        print(f"{results.names[int(class_id)],x1, y1, x2, y2}")
        

    #     if score > threshold:
    #         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
    #         cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    # out.write(frame)
    # ret, frame = cap.read()
    
    model.predict(video_path,show = True,conf=threshold)  #stream=True'


cap.release()
#out.release()
cv2.destroyAllWindows()

