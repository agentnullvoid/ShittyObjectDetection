import datetime
import numpy as np
import os
import sys
import tensorflow as tf
import cv2

from matplotlib import pyplot as plt

# Hack so that we include objection_detection dependencies in the path
sys.path.insert(0, '/ShittyObjectDetection/object_detection')

from utils import label_map_util

cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams

DETECTION_THRESHOLD = 0.5
PICTURE_DUMP_DIR = '/root/picturedump/'
MODEL_NAME = '/ShittyObjectDetection/ssd_inception_v2_coco_2017_11_17'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('/ShittyObjectDetection', 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

classes_to_detect = []
with open('/ShittyObjectDetection/classes_to_detect.txt', 'r') as fp:
    classes_to_detect = fp.readlines()


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            ret, image_np = cap.read()
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Check the detections for classes that we are interested in
            for index, value in enumerate(classes[0]):
                object_dict = {}
                if scores[0, index] > DETECTION_THRESHOLD:
                    category_name = (category_index.get(value)).get('name').encode('utf8')
                    # Change this conditional to include other classes
                    # Or be smart about it and pass some ENV or config file
                    if category_name in classes_to_detect:
                        capture_time = now.strftime("%Y-%m-%d_%H_%M_%S")
                        print("Detected [{}] on {} with probability: {}".format(category_name, capture_time, scores[0, index]))
                        now = datetime.datetime.now()
                        capture_name = capture_time + "_" + category_name + "_" + str(scores[0, index])
                        capture_file_dest = os.path.join(PICTURE_DUMP_DIR, capture_name + ".jpg")
                        cv2.imwrite(capture_file_dest, image_np)
