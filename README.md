Model and code imageDectection
---Raspberry pi4 Model B + Google coral USB----

<h1>-------การเปิดใช้งาน Environment----------------</h1>
*เปิด terminal*
cd Desktop
-------
# yolo

เปิด environment ของ yolo
source ~/yolo/bin/activate

# เปลี่ยน directory ไปยังตำแหน่งที่ต้องการ (yolo)
cd ~/laser-detect/yoloDetect/real

# รัน Python
python ... .py

-------
# tensorflow

เปิด environment ของ tensorflow
source ~/tflite/bin/activate

# เปลี่ยน directory ไปยังตำแหน่งที่ต้องการ (tensorflow)
cd ~/laser-detect/tensorflowDetect/real/

# รัน Python
python ... .py

-------------------------------------
วิธีสั่งให้ Google coral ทำงานสูงสุด (ยังไม่ได้ลอง กลัวไหม้)

**สามารถทำได้ใน terminal ของ Env ที่เปิดได้เลย

Install with maximum operating frequency (optional)
The above command installs the standard Edge TPU runtime for Linux, which operates the device at a reduced clock frequency. You can instead install a runtime version that operates at the maximum clock frequency. This increases the inferencing speed but also increases power consumption and causes the USB Accelerator to become very hot.

If you're not certain your application requires increased performance, you should use the reduced operating frequency. Otherwise, you can install the maximum frequency runtime as follows:

```sudo apt-get install libedgetpu1-max```

You cannot have both versions of the runtime installed at the same time, but you can switch by simply installing the alternate runtime as shown above.

Caution: When operating the device using the maximum clock frequency, the metal on the USB Accelerator can become very hot to the touch. This might cause burn injuries. To avoid injury, either keep the device out of reach when operating it at maximum frequency, or use the reduced clock frequency.

ref: https://coral.ai/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime


