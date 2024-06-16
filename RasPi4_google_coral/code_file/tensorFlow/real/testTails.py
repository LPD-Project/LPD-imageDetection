import cv2
import numpy as np
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
import collections

Object = collections.namedtuple('Object', ['label', 'score', 'bbox'])

def tiles_location_gen(img_size, tile_size, overlap):
    tile_width, tile_height = tile_size
    img_width, img_height = img_size
    h_stride = tile_height - overlap
    w_stride = tile_width - overlap
    for h in range(0, img_height, h_stride):
        for w in range(0, img_width, w_stride):
            xmin = w
            ymin = h
            xmax = min(img_width, w + tile_width)
            ymax = min(img_height, h + tile_height)
            yield [xmin, ymin, xmax, ymax]

def reposition_bounding_box(bbox, tile_location):
    bbox[0] = bbox[0] + tile_location[0]
    bbox[1] = bbox[1] + tile_location[1]
    bbox[2] = bbox[2] + tile_location[0]
    bbox[3] = bbox[3] + tile_location[1]
    return bbox

class Detector:
    def __init__(self, model_path, labels_path, top_k, camera_idx, threshold, tile_size, tile_overlap, iou_threshold):
        self.model_path = model_path
        self.labels_path = labels_path
        self.top_k = top_k
        self.camera_idx = camera_idx
        self.threshold = threshold
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.iou_threshold = iou_threshold

        print(f'Loading {self.model_path} with {self.labels_path} labels.')
        self.interpreter = make_interpreter(self.model_path)
        self.interpreter.allocate_tensors()
        self.labels = read_label_file(self.labels_path)
        self.inference_size = input_size(self.interpreter)

    def run(self):
        #cap = cv2.VideoCapture(self.camera_idx) 
        cap = cv2.VideoCapture("manyObject.mp4") #*****
        cap.set(3, 1920)
        cap.set(4, 1080)
        
        while cap.isOpened():
            start_tick = cv2.getTickCount()
            ret, frame = cap.read()
            if not ret:
                break

            objects_by_label = dict()
            img_size = (frame.shape[1], frame.shape[0])
            for tile_location in tiles_location_gen(img_size, self.tile_size, self.tile_overlap):
                tile = frame[tile_location[1]:tile_location[3], tile_location[0]:tile_location[2]]
                tile_resized = cv2.resize(tile, self.inference_size)
                tile_rgb = cv2.cvtColor(tile_resized, cv2.COLOR_BGR2RGB)
                run_inference(self.interpreter, tile_rgb.tobytes())
                objs = get_objects(self.interpreter, self.threshold)
                
                for obj in objs:
                    bbox = [obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax]
                    bbox = reposition_bounding_box(bbox, tile_location)
                    label = self.labels.get(obj.id, '')
                    objects_by_label.setdefault(label, []).append(Object(label, obj.score, bbox))

            for label, objects in objects_by_label.items():
                idxs = self.non_max_suppression(objects, self.iou_threshold)
                for idx in idxs:
                    obj = objects[idx]
                    print(obj.label)
                    cv2.rectangle(frame, (obj.bbox[0], obj.bbox[1]), (obj.bbox[2], obj.bbox[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f'{obj.label}: {obj.score:.2f}', (obj.bbox[0], obj.bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            end_tick = cv2.getTickCount()
            elapsed_time = (end_tick - start_tick) / cv2.getTickFrequency()
            fps = 1 / elapsed_time
            print(f"FPS: {fps:.2f}")
            cv2.namedWindow("FRAME", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("FRAME", 1024, 768)

            cv2.imshow('FRAME', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def non_max_suppression(self, objects, threshold):
        if len(objects) == 1:
            return [0]

        boxes = np.array([o.bbox for o in objects])
        xmins = boxes[:, 0]
        ymins = boxes[:, 1]
        xmaxs = boxes[:, 2]
        ymaxs = boxes[:, 3]

        areas = (xmaxs - xmins) * (ymaxs - ymins)
        scores = [o.score for o in objects]
        idxs = np.argsort(scores)

        selected_idxs = []
        while idxs.size != 0:
            selected_idx = idxs[-1]
            selected_idxs.append(selected_idx)

            overlapped_xmins = np.maximum(xmins[selected_idx], xmins[idxs[:-1]])
            overlapped_ymins = np.maximum(ymins[selected_idx], ymins[idxs[:-1]])
            overlapped_xmaxs = np.minimum(xmaxs[selected_idx], xmaxs[idxs[:-1]])
            overlapped_ymaxs = np.minimum(ymaxs[selected_idx], ymaxs[idxs[:-1]])

            w = np.maximum(0, overlapped_xmaxs - overlapped_xmins)
            h = np.maximum(0, overlapped_ymaxs - overlapped_ymins)

            intersections = w * h
            unions = areas[idxs[:-1]] + areas[selected_idx] - intersections
            ious = intersections / unions

            idxs = np.delete(
                idxs, np.concatenate(([len(idxs) - 1], np.where(ious > threshold)[0])))

        return selected_idxs
import os
default_model_dir = './models'
default_model = 'bestEF0v2_edgetpu.tflite'
default_labels = 'label.txt'
if __name__ == '__main__':
    detector = Detector(
        model_path=os.path.join(default_model_dir, default_model),
        labels_path=os.path.join(default_model_dir, default_labels),
        top_k=0, #*************
        camera_idx=0, #*************
        threshold=0.4, #*************
        tile_size=(224, 224), #*************
        tile_overlap=0, #*************
        iou_threshold=0.4 #*************
    )
    detector.run()
