import numpy as np
import cv2
import os
import sys
import glob
import random
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter

import matplotlib
import matplotlib.pyplot as plt
num_frames = 0
start_time = cv2.getTickCount()
#pd_model = "./pd2/custom_model_liteV2/custom_model_liteV2/detect.tflite"
pd_model = "./pd2/custom_model_lite/detect.tflite"

pd_vid = './pd/videos/Y2meta.mp4'

#pd_label = './pd2/custom_model_liteV2/custom_model_liteV2/labelmap.txt'
pd_label = './pd2/custom_model_lite/labelmap.txt'

modelpath=pd_model
lblpath=pd_label
min_conf=0.5
cap = cv2.VideoCapture(pd_vid)
num_threads = 1

interpreter = Interpreter(model_path=modelpath, num_threads=num_threads)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

float_input = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5\
    
    
fps_out = 0    

mouse_x, mouse_y = 0, 0
mouse_clicked = False
# Callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Store the mouse position in global variables
        global mouse_x, mouse_y
        mouse_x, mouse_y = x, y
        print(f"Mouse Position (X, Y): ({x}, {y})")
        global mouse_clicked
    if event == cv2.EVENT_LBUTTONDOWN:  # Check for left mouse button click
        mouse_clicked = not mouse_clicked    
    
cv2.namedWindow('Pigeon Detection')
cv2.setMouseCallback('Pigeon Detection', mouse_callback)    
    
import tensorflow as tf
tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
with open(lblpath, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
num_frames = 0
start_time = cv2.getTickCount()
while(True):
    ret, frame =cap.read()
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imH, imW, _ = frame.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std
        
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects
    
    detections = []
    
    
    for i in range(len(scores)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
          
                
            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])
    num_frames += 1
    if num_frames % 10 == 0:  # Calculate FPS every 10 frames
        end_time = cv2.getTickCount()
        elapsed_time = (end_time - start_time) / cv2.getTickFrequency()
        fps = 10 / elapsed_time
        print(f"FPS: {fps:.2f}")
        fps_out += fps
        if fps_out >= 100 :
            print(f"obj: {object_name},{xmin},{ymin}, {xmax},{ymax}")
            fps_out = 0   
            print(f"{detections}")
            
        start_time = cv2.getTickCount()
        
        
        
    
    mouse_position_text = f"Mouse Position (X, Y): ({mouse_x}, {mouse_y})"
    cv2.putText(frame, mouse_position_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Pigeon Detection',frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
    
cap.release()
cv2.destroyAllWindows()
    