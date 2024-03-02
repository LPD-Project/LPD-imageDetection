import cv2
from jetson_inference import detectNet
from jetson_utils import cudaFromNumpy, cudaToNumpy
from DetectedObjectList import DetectedObjectList
from MapObject import MapObject

import time

class ObjectDetector:
    
    
    
    def __init__(self, camera_pipeline, model_name="ssd-mobilenet-v2", threshold=0.5):
        self.net = detectNet(model_name, threshold=threshold)
        self.cap = cv2.VideoCapture(camera_pipeline, cv2.CAP_GSTREAMER)
        self.detectedList = DetectedObjectList()
        self.start = False
        
    def send_detected_objects(self):
        data_pack = []
        for index, obj in enumerate(self.detectedList, start=1):
            data = {
                "Object Number": index,
                "Top": obj.top,
                "Left": obj.left,
                "Bottom": obj.bottom,
                "Right": obj.right,
                "Class": obj.class_name,
            }
            data_pack.append(data)
        return data_pack
    
    @staticmethod
    def get_person_data(data_pos):
        person_data = []
        for obj_data in data_pos:
            if obj_data["Class"] == "person":
                person_data.append(obj_data)
        return person_data
    @staticmethod
    def interrupt_if_person_detected(data_pos):
        person_data = ObjectDetector.get_person_data(data_pos)
        if person_data:
            return person_data
        return None
        
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
                            
    def to_another(self,data_sender):
        print("send")
        person_data = self.interrupt_if_person_detected(data_sender)
        if person_data:
            print("Person detected! Sending person data:")
            for obj in person_data:
                print(obj)
        else:
            print("No person detected.")
        self.calPos(data_sender)

        
        #for obj_data in data_sender:
         #   print("Object Number:", obj_data["Object Number"])
         #   print("Top:", obj_data["Top"])
         #   print("Left:", obj_data["Left"])
         #   print("Bottom:", obj_data["Bottom"])
         #   print("Right:", obj_data["Right"])
         #   print("Class:", obj_data["Class"])
         #   print()

        
    def run(self):
        if not self.cap.isOpened():
            print("Error: Couldn't open camera.")
            exit()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Couldn't read frame.")
                break

            cuda_img = cudaFromNumpy(frame)
            detections = self.net.Detect(cuda_img)
            self.detectedList.clear()
            frame_bgr = cudaToNumpy(cuda_img)

     

            for detection in detections:
                left, top, right, bottom = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)
                cv2.rectangle(frame_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame_bgr, self.net.GetClassDesc(detection.ClassID), (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                obj = MapObject(left, top, right, bottom, self.net.GetClassDesc(detection.ClassID))
                self.detectedList.add_object(obj)
                
            if self.start:
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                #print(f"current_time : {current_time}")

                if elapsed_time >= 3:
                    self.start_time = time.time()
                    self.to_another(self.send_detected_objects())
                    #for index, obj in enumerate(self.detectedList, start=1):
                        #print(f"Object Number: {index}")
                        #print("Top:", obj.top)
                        #print("Left:", obj.left)
                        #print("Bottom:", obj.bottom)
                        #print("Right:", obj.right)
                        #print("Class:", obj.class_name)
                        #print(len(self.detectedList))
                        #print("self.detectedList",self.detectedList)
                        #print()
                        

            cv2.putText(frame_bgr, "{:.0f} FPS".format(self.net.GetNetworkFPS()), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Object Detection", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
            if not self.start:
                self.start = True
                self.start_time = time.time()
                print("out start")

        self.cap.release()
        cv2.destroyAllWindows()