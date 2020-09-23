#!/usr/bin/python3
# -*- coding: utf-8 -*-
import cv2
import os
import time


class general_faster_rcnn(object):
    def __init__(self, modelpath):
        self.conf_threshold = 0.3  # Confidence threshold
        self.nms_threshold = 0.4  # Non-maximum suppression threshold
        self.net_width = 416  # 300 # Width of network's input image
        self.net_height = 416  # 300 # Height of network's input image

        self.classes = self.get_coco_names()
        self.faster_rcnn_model = self.get_faster_rcnn_model(modelpath)
        self.outputs_names = self.get_outputs_names()

    def get_coco_names(self):
        classes = ["person", "bicycle", "car", "motorcycle", "airplane",
                   "bus", "train", "truck", "boat", "traffic light",
                   "fire hydrant", "background", "stop sign", "parking meter",
                   "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                   "elephant", "bear", "zebra", "giraffe", "background",
                   "backpack", "umbrella", "background", "background",
                   "handbag", "tie", "suitcase", "frisbee", "skis",
                   "snowboard", "sports ball", "kite", "baseball bat",
                   "baseball glove", "skateboard", "surfboard", "tennis racket",
                   "bottle", "background", "wine glass", "cup", "fork", "knife",
                   "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                   "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                   "chair", "couch", "potted plant", "bed", "background",
                   "dining table", "background", "background", "toilet",
                   "background", "tv", "laptop", "mouse", "remote", "keyboard",
                   "cell phone", "microwave", "oven", "toaster", "sink",
                   "refrigerator", "background", "book", "clock", "vase",
                   "scissors", "teddy bear", "hair drier", "toothbrush",
                   "background"]

        return classes

    def get_faster_rcnn_model(self, modelpath):
        pb_file = os.path.join(modelpath, "frozen_inference_graph.pb")
        pbtxt_file = os.path.join(modelpath, "graph.pbtxt")

        net = cv2.dnn.readNetFromTensorflow(pb_file, pbtxt_file)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        return net

    def get_outputs_names(self):
        # 网络中所有网络层的名字
        layersNames = self.faster_rcnn_model.getLayerNames()
        # 网络输出层的名字，如，没有链接输出的网络层.

        return [layersNames[i[0] - 1] for i in \
                self.faster_rcnn_model.getUnconnectedOutLayers()]

    # NMS 处理掉低 confidence 的边界框.
    def postprocess(self, img_cv2, outputs):
        img_height, img_width, _ = img_cv2.shape

        class_ids = []
        confidences = []
        boxes = []
        for output in outputs:
            for detection in output[0, 0]:
                # [batch_id, class_id, confidence, left, top, right, bottom]
                confidence = detection[2]
                if confidence > self.conf_threshold:
                    left = int(detection[3] * img_width)
                    top = int(detection[4] * img_height)
                    right = int(detection[5] * img_width)
                    bottom = int(detection[6] * img_height)
                    width = right - left + 1
                    height = bottom - top + 1

                    class_ids.append(int(detection[1]))
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # NMS 处理
        indices = cv2.dnn.NMSBoxes(boxes,
                                   confidences,
                                   self.conf_threshold,
                                   self.nms_threshold)

        results = []
        for ind in indices:
            res_box = {}
            res_box["class_id"] = class_ids[ind[0]]
            res_box["score"] = confidences[ind[0]]

            box = boxes[ind[0]]
            res_box["box"] = (box[0], box[1], box[0] + box[2], box[1] + box[3])

            results.append(res_box)

        return results

    def predict(self, img_file):
        img_cv2 = cv2.imread(img_file)

        # 创建 4D blob.
        blob = cv2.dnn.blobFromImage(
            img_cv2,
            size=(self.net_width, self.net_height),
            swapRB=True, crop=False)

        # 设置网络的输入 blob
        self.faster_rcnn_model.setInput(blob)

        # 打印网络的输出层名
        print("[INFO]Net output layers: {}".format(self.outputs_names))

        # Runs forward
        outputs = self.faster_rcnn_model.forward(self.outputs_names)

        # NMS
        results = self.postprocess(img_cv2, outputs)

        return results

    def vis_res(self, img_file, results):
        img_cv2 = cv2.imread(img_file)

        for result in results:
            left, top, right, bottom = result["box"]
            cv2.rectangle(img_cv2,
                          (left, top),
                          (right, bottom),
                          (255, 178, 50), 3)

            # Get the label for the class name and its confidence
            label = '%.2f' % result["score"]
            if self.classes:
                assert (result["class_id"] < len(self.classes))
                label = '%s:%s' % (self.classes[result["class_id"]], label)

            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, label_size[1])
            cv2.rectangle(
                img_cv2,
                (left, top - round(1.5 * label_size[1])),
                (left + round(1.5 * label_size[0]), top + baseline),
                (255, 0, 0),
                cv2.FILLED)
            cv2.putText(img_cv2,
                        label,
                        (left, top),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), 1)

        t, _ = self.faster_rcnn_model.getPerfProfile()
        label = 'Inference time: %.2f ms' % \
                (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(img_cv2, label, (0, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        cv2.namedWindow('OpenCV DNN Faster RCNN-inception_v2', cv2.WINDOW_NORMAL)
        cv2.imshow('OpenCV DNN Faster RCNN-inception_v2', img_cv2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    print("[INFO]Faster RCNN object detection in OpenCV.")

    img_file = "dog.jpg"

    start = time.time()
    modelpath = "faster_rcnn_inception_v2_coco_2018_01_28"
    faster_rcnn_model = general_faster_rcnn(modelpath)
    print("[INFO]Model loads time: ", time.time() - start)

    start = time.time()
    results = faster_rcnn_model.predict(img_file)
    print("[INFO]Model predicts time: ", time.time() - start)
    faster_rcnn_model.vis_res(img_file, results)
