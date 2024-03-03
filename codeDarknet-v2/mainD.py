import argparse
from DarknetObjectDetector import*


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)960, height=(int)540, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink",
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="./LPD-imageDetection/yoloModel-temp/yolov4-tiny-1/v1/custom-yolov4-tiny-detector_best.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action="store_true",
                        help="window inference display. For headless systems")
    parser.add_argument("--ext_output", action="store_true",
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./LPD-imageDetection/yoloModel-temp/yolov4-tiny-1/v1/custom-yolov4-tiny-detector.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./LPD-imageDetection/yoloModel-temp/yolov4-tiny-1/v1/obj.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()

if __name__ == "__main__":
    args = parser()

    # สร้าง ObjectDetector
    darknet_detector = DarknetObjectDetector(
        input_path=args.input,
        weights=args.weights,
        config_file=args.config_file,
        data_file=args.data_file,
        thresh=args.thresh,
        out_filename=args.out_filename
    )

    # เริ่มการทำงานของ ObjectDetector
    darknet_detector.run()
