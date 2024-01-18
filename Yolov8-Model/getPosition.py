from ultralytics import YOLO

import os
import cv2
from ultralytics.utils.plotting import Annotator
import numpy as np 
import time 

model_path = os.path.join('.', 'Model', 'bestV7.pt')
VIDEOS_DIR = os.path.join('.', 'videos')
IMG_DIR = os.path.join('.', 'img')
video_path = os.path.join(VIDEOS_DIR, 'Video_TEST3.mp4')

img_path = os.path.join(IMG_DIR, 'pigeon-vid.jpg')
model = YOLO(model_path)
cap = cv2.VideoCapture(img_path)

prev_frame_time = 0
new_frame_time = 0

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

# Create a separate window for mouse position
cv2.namedWindow('Pigeon Detection')
cv2.setMouseCallback('Pigeon Detection', mouse_callback)

# output_file = 'output_video.mp4'
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')   
# fps_out = 30.0 
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out = cv2.VideoWriter(output_file, fourcc, 30, (frame_width,frame_height))

start = False
while True:
    ret, img = cap.read()
    
    if not ret: 
        break
    
    # if mouse_clicked :
    #     print("click")
    #     while mouse_clicked :
    #         if cv2.waitKey(1) & 0xFF == ord(' '):
    #             break
    
    
    results = model.predict(img)
    
    

    for r in results:
        annotator = Annotator(img,font_size=0.1)
        
        boxes = r.boxes
        for box in boxes:
            print(f"boxSSSS: {box}")
            print(f"resultsTypeTTTTTTTTTTTTTTTTTTTT: {type(box)}")
            # box คือ แต่ละ กรอบ ที่มันจับได้ ใน boxes(ใหญ่)
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            Top = int(b[0])
            Left = int(b[1])
            Bottom = int(b[2])
            Right = int(b[3])
            h = Right - Left  # wide
            w = Bottom - Top  # high
            
            
            c = box.cls
            #annotator.box_label(b)
            

            annotator.box_label(b, model.names[int(c)])
            
            
            print(f"{model.names[int(c)]},Top-Left: {(Top,Left)}, Bottom-Right: {(Bottom,Right)}")
            box_position_text1 = f"{Top,Left}" #tl
            box_position_text2 = f"{Bottom,Right}" #br
            box_position_text3 = f"{Top+w,Left}" #tr
            box_position_text4 = f"{Bottom-w,Right}" #bl
            
            
            cv2.putText(img, box_position_text1, (Top-70, Left+3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(img, box_position_text2, (Bottom+1, Right-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(img, box_position_text3, ((Top+w)+1, Left+3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(img, box_position_text4, ((Bottom-w)-70, Right-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            
           
            
            
           
            
          
    img = annotator.result()
    
    
    
    # Draw a circle at the mouse position
    # if 'mouse_x' in globals() and 'mouse_y' in globals():
    #     cv2.circle(img, (mouse_x, mouse_y), 1, (0, 0, 255), -1)
    
    #cv2.imshow('Pigeon Detection', img)
    mouse_position_text = f"Mouse Position (X, Y): ({mouse_x}, {mouse_y})"
    cv2.putText(img, mouse_position_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time
     # converting the fps into integer 
    fps = int(fps) 
  
    # converting the fps to string so that we can display it on frame 
    # by using putText function 
    fps_list = []
    fps = int(fps)
    fps_list.append(fps)
    average_fps = sum(fps_list) / len(fps_list)
    FPS_text = f"FPS: {average_fps}" 
    
    
    cv2.putText(img, FPS_text, (18, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    
     
    cv2.imshow('Pigeon Detection', img)  # Show mouse position window
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
    # out.write(img)
    if start == False:
         time.sleep(10)
         start = True
   
# out.release() 
cap.release()

cv2.destroyAllWindows()


