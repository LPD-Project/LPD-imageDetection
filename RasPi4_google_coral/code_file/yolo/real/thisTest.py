import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
import time

model = YOLO('./models/yolov4_512_full_integer_quant_edgetpu.tflite')

cap = cv2.VideoCapture("manyObject.mp4")
cap.set(3, 1920)
cap.set(4, 1080)

my_file = open("./models/label.txt", "r")
data = my_file.read()
class_list = data.split("\n")

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    results = model.predict(frame, imgsz=512 ,iou = 0)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
#แจกแจงวัตถุทีละอัน
    for index, row in px.iterrows():   
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

    # Calculate FPS
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time

    # Display FPS on frame
    cvzone.putTextRect(frame, f'FPS: {round(fps, 2)}', (10, 30), 1,1)
    cv2.namedWindow("FRAME", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("FRAME", 800, 600) 
    cv2.imshow("FRAME", frame)
    print(f'FPS: {round(fps, 2)}')

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
