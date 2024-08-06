import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
import time
import tkinter as tk

class ModelSwitcherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Model Switcher")

        # ปุ่มสำหรับสลับโมเดล
        self.model_button1 = tk.Button(root, text="Use best.pt", command=self.switch_to_best)
        self.model_button1.pack()

        self.model_button2 = tk.Button(root, text="Use yolov8m.pt", command=self.switch_to_yolov8m)
        self.model_button2.pack()

        self.label = tk.Label(root, text="Current Model: yolov8m.pt")
        self.label.pack()

        # ตัวแปรโมเดลและไฟล์คลาส
        self.model_path = './yolov8m.pt'
        self.class_file = "coco.txt"
        self.model = YOLO(self.model_path)

        # การเตรียมกล้องหรือวิดีโอ
        self.cap = cv2.VideoCapture("2024-06-14 16-41-36.mp4")
        self.cap.set(3, 1920)
        self.cap.set(4, 1080)
        
        # โหลดไฟล์คลาสเริ่มต้น
        self.load_class_file()

        self.frame_count = 0
        self.start_time = time.time()

        # เริ่มการแสดงผลวิดีโอ
        self.update_video()

    def load_class_file(self):
        with open(self.class_file, "r") as f:
            data = f.read()
        self.class_list = data.split("\n")

    def switch_to_best(self):
        self.model_path = './version5/weights/best.pt'
        self.class_file = "label.txt"
        self.model = YOLO(self.model_path)
        self.load_class_file()
        self.label.config(text="Current Model: best.pt")

    def switch_to_yolov8m(self):
        self.model_path = './yolov8m.pt'
        self.class_file = "coco.txt"
        self.model = YOLO(self.model_path)
        self.load_class_file()
        self.label.config(text="Current Model: yolov8m.pt")

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            return

        self.frame_count += 1
        if self.frame_count % 3 == 0:
            results = self.model.predict(frame, imgsz=1024, conf=0.35)
            a = results[0].boxes.data
            px = pd.DataFrame(a).astype("float")
            for index, row in px.iterrows():
                x1, y1, x2, y2, p, d = int(row[0]), int(row[1]), int(row[2]), int(row[3]), float(row[4]), int(row[5])
                c = self.class_list[d]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{c} {round(p, 2)}', (x1, y1), 1, 1)

        # Calculate FPS
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        fps = self.frame_count / elapsed_time

        # Display FPS on frame
        cvzone.putTextRect(frame, f'FPS: {round(fps, 2)}', (10, 30), 1, 1)
        cv2.namedWindow("FRAME", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("FRAME", 1920, 1080)
        cv2.imshow("FRAME", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            self.cap.release()
            cv2.destroyAllWindows()
            self.root.quit()
            return

        self.root.after(1, self.update_video)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelSwitcherApp(root)
    root.mainloop()
