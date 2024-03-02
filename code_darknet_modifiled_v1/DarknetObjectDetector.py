import darknet
import cv2
import time
import threading
import queue

class DarknetObjectDetector:
    def __init__(self, input_path, weights, config_file, data_file, thresh=0.25, out_filename=""):
        self.input_path = input_path
        self.weights = weights
        self.config_file = config_file
        self.data_file = data_file
        self.thresh = thresh
        self.out_filename = out_filename
        self.stop_flag = threading.Event()

    def video_capture(self, raw_frame_queue, preprocessed_frame_queue, darknet_height, darknet_width):
        cap = cv2.VideoCapture(self.input_path)
        while cap.isOpened() and not self.stop_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)
            raw_frame_queue.put(frame)
            img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
            darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
            preprocessed_frame_queue.put(img_for_detect)
        self.stop_flag.set()
        cap.release()

    def inference(self, preprocessed_frame_queue, detections_queue, network, class_names):
        while not self.stop_flag.is_set():
            darknet_image = preprocessed_frame_queue.get()
            prev_time = time.time()
            detections = darknet.detect_image(network, class_names, darknet_image, thresh=self.thresh)
            detections_queue.put(detections)
            darknet.free_image(darknet_image)

    def drawing(self, raw_frame_queue, preprocessed_frame_queue, detections_queue, fps_queue,
            darknet_height, darknet_width, vid_h, vid_w, class_colors):
    while not self.stop_flag.is_set():
        frame = raw_frame_queue.get()
        detections = detections_queue.get()

        detections_adjusted = []
        if frame is not None:
            for label, confidence, bbox in detections:
                # ปรับขนาดตำแหน่งของ bbox ให้เหมาะสมกับภาพเดิม
                bbox_adjusted = convert2original(frame, bbox, darknet_height, darknet_width)
                # เพิ่มข้อมูลตรวจจับที่ปรับแล้วเข้าไปใน detections_adjusted
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
            # วาดกรอบและข้อความบนภาพ
            image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
            # แสดงภาพผ่านทางหน้าต่าง
            cv2.imshow("Inference", image)
            # หากมีการกดปุ่ม ESC ให้ออกจาก loop
            if cv2.waitKey(1) == 27:
                break
    cv2.destroyAllWindows()


    def run(self):
        darknet_width = ...
        darknet_height = ...
        cap = cv2.VideoCapture(self.input_path)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        del cap

        raw_frame_queue = queue.Queue()
        preprocessed_frame_queue = queue.Queue(maxsize=1)
        detections_queue = queue.Queue(maxsize=1)
        fps_queue = queue.Queue(maxsize=1)

        network, class_names, class_colors = darknet.load_network(
            self.config_file,
            self.data_file,
            self.weights,
            batch_size=1)

        exec_units = (
            threading.Thread(target=self.video_capture,
                             args=(raw_frame_queue, preprocessed_frame_queue, darknet_height, darknet_width)),
            threading.Thread(target=self.inference,
                             args=(preprocessed_frame_queue, detections_queue, network, class_names)),
            threading.Thread(target=self.drawing,
                             args=(raw_frame_queue, preprocessed_frame_queue, detections_queue, fps_queue,
                                   darknet_height, darknet_width, video_height, video_width, class_colors)),
        )
        for exec_unit in exec_units:
            exec_unit.start()
        for exec_unit in exec_units:
            exec_unit.join()

        print("\nDone.")
