import cv2
from jetson_inference import detectNet
from jetson_utils import cudaFromNumpy, cudaToNumpy
from DetectedObjectList import *
from MapObject import *


# Load the detection network
net = detectNet("ssd-mobilenet-v2", threshold=0.5)

# Open the CSI camera
def gstreamer_pipeline(
    capture_width=960,
    capture_height=540,
    display_width=1920,
    display_height=1080,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int){}, height=(int){}, framerate=(fraction){}/1 ! "
        "nvvidconv flip-method={} ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        .format(capture_width, capture_height, framerate, flip_method)
    )

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

detectedList = DetectedObjectList()





# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Couldn't open camera.")
    exit()

while True:
    # Read frame from the camera
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Couldn't read frame.")
        break

    # Perform object detection on the frame
    cuda_img = cudaFromNumpy(frame)
    detections = net.Detect(cuda_img)
    detectedList.clear()
    # Convert cudaImage back to numpy array for visualization
    frame_bgr = cudaToNumpy(cuda_img)

    # Draw bounding boxes and labels on the frame
    for detection in detections:
        left, top, right, bottom = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)
        
        
        cv2.rectangle(frame_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame_bgr, net.GetClassDesc(detection.ClassID), (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #print(detections[0].ClassID)
        #print(detections)
        #print(net.GetClassDesc(detection.ClassID))
        obj = MapObject(left, top, right, bottom,net.GetClassDesc(detection.ClassID))
        detectedList.add_object(obj)
        print(len(detectedList))

        
    # Display the frame with FPS
    cv2.putText(frame_bgr, "{:.0f} FPS".format(net.GetNetworkFPS()), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Object Detection", frame_bgr)
    #print("{:.0f} FPS".format(net.GetNetworkFPS()))

    # Check for key press and break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
