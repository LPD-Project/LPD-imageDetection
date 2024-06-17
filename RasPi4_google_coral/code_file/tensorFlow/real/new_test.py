import argparse
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

def visualize(image, detection_result):
    for detection in detection_result.detections:
        # วาดกรอบสี่เหลี่ยมรอบวัตถุ
        bbox = detection.bounding_box
        start_point = (int(bbox.origin_x), int(bbox.origin_y))
        end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
        cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
        
        # วาดชื่อวัตถุและคะแนนความเชื่อมั่น
        category = detection.categories[0]
        class_name = category.category_name
        score = round(category.score, 2)
        result_text = f'{class_name}: {score}'
        text_location = (int(bbox.origin_x), int(bbox.origin_y - 10))
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    return image

def run(model: str, camera_id: str, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """รันการตรวจจับวัตถุอย่างต่อเนื่องบนภาพที่ได้จากกล้อง

  Args:
    model: ชื่อของโมเดล TFLite สำหรับการตรวจจับวัตถุ
    camera_id: ID ของกล้องที่จะใช้กับ OpenCV
    width: ความกว้างของเฟรมที่จับจากกล้อง
    height: ความสูงของเฟรมที่จับจากกล้อง
    num_threads: จำนวน threads ของ CPU ที่จะใช้รันโมเดล
    enable_edgetpu: True/False ระบุว่าโมเดลนี้ใช้ EdgeTPU หรือไม่
  """

  # ตัวแปรสำหรับคำนวณ FPS
  counter, fps = 0, 0
  start_time = time.time()

  # เริ่มจับภาพวิดีโอจากกล้อง
  cap = cv2.VideoCapture("manyObject.mp4") #---------------------
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

  # พารามิเตอร์สำหรับการแสดงผล
  row_size = 20  # พิกเซล
  left_margin = 24  # พิกเซล
  text_color = (0, 0, 255)  # สีแดง
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # เริ่มต้นโมเดลการตรวจจับวัตถุ
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=120, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  # จับภาพจากกล้องและรันการตรวจจับวัตถุอย่างต่อเนื่อง
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: ไม่สามารถอ่านจากกล้องได้ กรุณาตรวจสอบการตั้งค่ากล้อง.'
      )

    counter += 1
    # image = cv2.flip(image, 1)
    # แปลงภาพจาก BGR เป็น RGB ตามที่โมเดล TFLite ต้องการ
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # สร้างวัตถุ TensorImage จากภาพ RGB
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # รันการตรวจจับวัตถุด้วยโมเดล
    detection_result = detector.detect(input_tensor)

    # วาดกรอบและข้อมูลบนภาพ
    image = visualize(image, detection_result)

    # คำนวณค่า FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # แสดงค่า FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    # text_location = (left_margin, row_size)
    # cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
    #             font_size, text_color, font_thickness)
    print(f"FPS: {fps_text}")

    # หยุดโปรแกรมหากกดปุ่ม ESC
    if cv2.waitKey(1) == 27:
      break
    cv2.namedWindow("object_detector", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("object_detector", 1024, 768)
    cv2.imshow('object_detector', image)

  cap.release()
  cv2.destroyAllWindows()

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path ของโมเดลการตรวจจับวัตถุ.',
      required=False,
      default='./models/bestEF0v2_edgetpu.tflite') #--------
  parser.add_argument(
      '--cameraId', help='ID ของกล้อง.', required=False, default=0)
  parser.add_argument(
      '--frameWidth',
      help='ความกว้างของเฟรมที่จะจับจากกล้อง.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='ความสูงของเฟรมที่จะจับจากกล้อง.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='จำนวน threads ของ CPU ที่จะใช้รันโมเดล.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='ระบุว่าจะรันโมเดลบน EdgeTPU หรือไม่.',
      action='store_true',
      required=False,
      default=True)
  args = parser.parse_args()

  run(args.model, str(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))

if __name__ == '__main__':
  main()
