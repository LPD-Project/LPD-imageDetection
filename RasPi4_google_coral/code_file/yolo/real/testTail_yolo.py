import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import time
import numpy as np

# Define tiles_location_gen and reposition_bounding_box functions
def tiles_location_gen(img_size, tile_size, overlap):
    tile_width, tile_height = tile_size
    img_width, img_height = img_size
    h_stride = tile_height - overlap
    w_stride = tile_width - overlap
    for h in range(0, img_height, h_stride):
        for w in range(0, img_width, w_stride):
            xmin = w
            ymin = h
            xmax = min(img_width, w + tile_width)
            ymax = min(img_height, h + tile_height)
            yield [xmin, ymin, xmax, ymax]

def reposition_bounding_box(bbox, tile_location):
    bbox[0] = bbox[0] + tile_location[0]
    bbox[1] = bbox[1] + tile_location[1]
    bbox[2] = bbox[2] + tile_location[0]
    bbox[3] = bbox[3] + tile_location[1]
    return bbox

model = YOLO('./models/yolov4_512_full_integer_quant_edgetpu.tflite')

cap = cv2.VideoCapture("manyObject.mp4")
cap.set(3, 1920)
cap.set(4, 1080)

my_file = open("./models/label.txt", "r")
data = my_file.read()
class_list = data.split("\n")

frame_count = 0
start_time = time.time()

tile_size = (1024, 1024)  # Define your tile size
tile_overlap = 0  # Define your tile overlap

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    img_size = (frame.shape[1], frame.shape[0])
    objects_by_label = dict()

    for tile_location in tiles_location_gen(img_size, tile_size, tile_overlap):
        tile = frame[tile_location[1]:tile_location[3], tile_location[0]:tile_location[2]]
        results = model.predict(tile, imgsz=512, iou=0.2)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]

            bbox = [x1, y1, x2, y2]
            bbox = reposition_bounding_box(bbox, tile_location)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{c}', (bbox[0], bbox[1]), 1, 1)

    # Calculate FPS
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time

    # Display FPS on frame
    cvzone.putTextRect(frame, f'FPS: {round(fps, 2)}', (10, 30), 1, 1)
    cv2.namedWindow("FRAME", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FRAME", 800, 600)
    cv2.imshow("FRAME", frame)
    print(f'FPS: {round(fps, 2)}')

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()