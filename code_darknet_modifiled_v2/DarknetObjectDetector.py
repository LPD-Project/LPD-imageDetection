import darknet
import cv2
import time
import threading
import queue
import subprocess
from DetectedObjectList import*
from MapObject import*

class DarknetObjectDetector:
    def __init__(self, input_path, weights, config_file, data_file, thresh=0.6, out_filename=""):
	
        self.input_path = input_path
        self.weights = weights
        self.config_file = config_file
        self.data_file = data_file
        self.thresh = thresh
        self.out_filename = out_filename
        self.stop_flag = threading.Event()
	    self.detected_objects = DetectedObjectList()  # สร้าง DetectedObjectList

    @staticmethod
    def restart_nvargus_daemon():
    	try:
            # เรียกใช้ subprocess เพื่อรันคำสั่ง sudo service nvargus-daemon restart
            subprocess.run(['sudo', 'service', 'nvargus-daemon', 'restart'], check=True)
            print("Restarted nvargus-daemon service successfully.")
    	except subprocess.CalledProcessError as e:
            print("Error restarting nvargus-daemon service:", e)

    def to_another(self,detected_objects):
        self.calPos(detected_objects) #ส่งนกที่ใกล้ที่สุด
        # ทำงานที่ต้องการทำกับ detected_objects ที่ได้รับเข้ามา
        pass
    
    @staticmethod        
    def calPos(data_pos):
        class_name = "bird"
        closest_obj = None
        closest_distance = float('inf')

       

        for obj_data in data_pos:
                if obj_data["Class"] != class_name:
                    continue  # ข้าม obj_data ที่ไม่มีคลาสเป็น "bird"

                bottom_distance = abs(obj_data["Bottom"] - 540)
                left_distance = abs(obj_data["Left"])
                
                if left_distance < closest_distance and bottom_distance < closest_distance:
                    closest_obj = obj_data
                    closest_distance = max(left_distance, bottom_distance)

        if closest_obj is not None:
                print("Object Number:", closest_obj["Object Number"])
                print("Top:", closest_obj["Top"])
                print("Left:", closest_obj["Left"])
                print("Bottom:", closest_obj["Bottom"])
                print("Right:", closest_obj["Right"])
                print("Class:", closest_obj["Class"])
                return closest_obj
        else:
                print("No bird object found meeting the criteria.")
                return None

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

    @staticmethod
    def convert2relative(bbox, preproc_h, preproc_w):
        """
        YOLO format use relative coordinates for annotation
        """
        x, y, w, h = bbox
        return x / preproc_w, y / preproc_h, w / preproc_w, h / preproc_h

    @staticmethod
    def convert2original(image, bbox, preproc_h, preproc_w):
    	x, y, w, h = DarknetObjectDetector.convert2relative(bbox, preproc_h, preproc_w)


    	image_h, image_w, __ = image.shape

    	orig_x = int(x * image_w)
    	orig_y = int(y * image_h)
    	orig_width = int(w * image_w)
    	orig_height = int(h * image_h)

    	bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    	return bbox_converted	

    def drawing(self, raw_frame_queue, preprocessed_frame_queue, detections_queue, fps_queue,
                darknet_height, darknet_width, vid_h, vid_w, class_colors):
        while not self.stop_flag.is_set():
            frame = raw_frame_queue.get()
            detections = detections_queue.get()

            detections_adjusted = []
            if frame is not None and detections:
                for label, confidence, bbox in detections:
                    # ปรับขนาดตำแหน่งของ bbox ให้เหมาะสมกับภาพเดิม
                    bbox_adjusted = self.convert2original(frame, bbox, darknet_height, darknet_width)
                    # เพิ่ม MapObject เข้าใน DetectedObjectList
                    obj = MapObject(label, float(confidence), *bbox_adjusted)
                    self.detected_objects.add_object(obj)
                    
                    # เพิ่มข้อมูลตรวจจับที่ปรับแล้วเข้าไปใน detections_adjusted
                    detections_adjusted.append((str(label), confidence, bbox_adjusted))
            else:
                print('Not Found')

            # วาดกรอบและข้อความบนภาพ
            image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
            print(detections_adjusted)
	    detections_adjusted = []
            self.to_another(self.detected_objects)
	    self.detected_objects.clear()
            # แสดงภาพผ่านทางหน้าต่าง
            cv2.imshow("Inference", image)
            # หากมีการกดปุ่ม ESC ให้ออกจาก loop
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()



    def run(self):
        self.restart_nvargus_daemon()
        darknet_width = 640
        darknet_height = 480
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