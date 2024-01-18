Model and code imageDectection <br>
ไฟล์โมเดล จะอยู่ใน pd ในแต่ละเวอร์ชั่น ซึ่งเวอร์ชั่นที่ใช้ได้ตอนนี้คือ pd3 แต่ confiden ค่อนข้างต่ำ <br>
และนี่คือตัวอย่างการเซ็ท path ที่ถูกต้อง <br>
pd_model = "./pd3/custom_model_lite/detect.tflite" <br>
pd_label = "./pd3/custom_model_lite/labelmap.txt" <br>
ในส่วนของ pd_vid จะเป็นการระบุวีดีโอเพื่อใช้กับโมเดล ในกรณีที่ไม่อยากเปิดกล้อง <br>
โดยสามารถเซ็ท path ของ VDO ที่ pd_vid แล้วนำมาใส่ไว้ที่ <br>
cap = cv2.VideoCapture()  จะเป็น cap = cv2.VideoCapture(pd_vid)  <br>
ถ้าหาก จะเปิดกล้อง เพียงแค่เซ็ท <br>
cap = cv2.VideoCapture(0)  <br>
โดย 0 คือ ไดร์เวอร์ตัวแรกที่คอมเราตรวจจับได้ (หากมีหลายกล้องสามารถเปลี่ยนเป็น 1 หรือ 2 ...) <br>
