# Utilities for object detector.

import numpy as np
import sys
import tensorflow as tf
# from vehicle.hand_detection import Ui_MainWindow
import os
from threading import Thread
from datetime import datetime
import cv2
from utils import label_map_util
from collections import defaultdict
from utils import alertcheck
detection_graph = tf.Graph()
from object_detection.utils import ops as utils_ops
TRAINED_MODEL_DIR = 'frozen_graphs'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = TRAINED_MODEL_DIR + '/coco_frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = TRAINED_MODEL_DIR + '/mscoco_label_map.pbtxt'

NUM_CLASSES = 90
# load label map using utils provided by tensorflow object detection api
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

a=b=0

# Load a frozen infrerence graph into memory
def load_inference_graph():

    # load frozen tensorflow model into memory
    
    print("> ====== Loading frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Inference graph loaded.")
    return detection_graph, sess


def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, classes, im_width, im_height, image_np,Line_Position2,Orientation):
    # Determined using a piece of paper of known length, code can be found in distance to camera
    # print(image_np,"qqqqq")
    focalLength = 875
    # The average width of a human hand (inches) http://www.theaveragebody.com/average_hand_size.php
    # added an inch since thumb is not included
    avg_width = 4.0
    # To more easily differetiate distances and detected bboxes

    global a,b
    hand_cnt=0
    color = None
    color0 = (255,0,0)
    color1 = (0,50,255)
    for i in range(num_hands_detect):
        # print(a,b)
        
        if (scores[i] > score_thresh):
            print(scores[i])
        
            #no_of_times_hands_detected+=1
            #b=b+1
            #b=1
            #print(b)
            if classes[i] not in [3,6]:
                continue
            if classes[i] == 3:
                id = 'car'
                #b=1
            #
            # if classes[i] == 4:
            #     id ='motorcycle'
            #     avg_width = 3.0 # To compensate bbox size change
            #     #b=1

            if classes[i] == 6:
                id = 'bus'
                #b=1

            # if classes[i] == 8:
            #     id = 'truck'
            #     #b=1
            
            # if i == 0: color = color0
            # else:
            color = color1
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            # print(p1,p2)

            # dist = distance_to_camera(avg_width, focalLength, int(right-left))
            #
            # if dist:
            #     hand_cnt=hand_cnt+1
            cv2.rectangle(image_np, p1, p2, color , 3, 1)
            

            cv2.putText(image_np, id, (int(left), int(top)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , color, 2)

            cv2.putText(image_np, 'confidence: '+str("{0:.2f}".format(scores[i])),
                        (int(left),int(top)-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # cv2.putText(image_np, 'distance from camera: '+str("{0:.2f}".format(dist)+' inches'),
            #             (int(im_width*0.65),int(im_height*0.9+30*i)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2)
           
            # a=alertcheck.drawboxtosafeline(image_np,p1,p2,Line_Position2,Orientation)

            # if p2[1]>500 and classes[i] == 3 :
            #     print("car",a)
            a= np.count_nonzero(classes ==3)
                # Ui_MainWindow.setupUi.lineEdit_2.setText(str(a))
                #print(" no hand")
            # else:
            b= np.count_nonzero(classes ==6)
                # Ui_MainWindow.setupUi.lineEdit_3.setText(str(b))
                #print(" hand")
            
    return a,b

# Show fps value on image.
def draw_text_on_image(fps, image_np):
    print("ffgg")
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
# compute and return the distance from the hand to the camera using triangle similarity
def distance_to_camera(knownWidth, focalLength, pixelWidth):
    return (knownWidth * focalLength) / pixelWidth

# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)

# def run_inference_for_single_image(image, graph):
#   with graph.as_default():
#     with tf.Session() as sess:
#       # Get handles to input and output tensors
#       ops = tf.get_default_graph().get_operations()
#       all_tensor_names = {output.name for op in ops for output in op.outputs}
#       tensor_dict = {}
#       for key in [
#           'num_detections', 'detection_boxes', 'detection_scores',
#           'detection_classes', 'detection_masks'
#       ]:
#         tensor_name = key + ':0'
#         if tensor_name in all_tensor_names:
#           tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
#               tensor_name)
#       if 'detection_masks' in tensor_dict:
#         # The following processing is only for single image
#         detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
#         detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
#         # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
#         real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
#         detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
#         detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
#         detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
#             detection_masks, detection_boxes, image.shape[0], image.shape[1])
#         detection_masks_reframed = tf.cast(
#             tf.greater(detection_masks_reframed, 0.5), tf.uint8)
#         # Follow the convention by adding back the batch dimension
#         tensor_dict['detection_masks'] = tf.expand_dims(
#             detection_masks_reframed, 0)
#       image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
#       # print(image_tensor)
#
#       # Run inference
#       output_dict = sess.run(tensor_dict,
#                              feed_dict={image_tensor: image})
#
#   #     image_np_expanded = np.expand_dims(image, axis=0)
#   #
#   #
#   #     (boxes, scores, classes, num) = sess.run(
#   #                 [tensor_dict['detection_boxes'], tensor_dict['detection_scores'],
#   #                     tensor_dict['detection_scores'], tensor_dict['num_detections']],
#   #                 feed_dict={image_tensor: image_np_expanded})
#   # return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)
#
#
#       # all outputs are float32 numpy arrays, so convert types as appropriate
#       output_dict['num_detections'] = int(output_dict['num_detections'][0])
#       output_dict['detection_classes'] = output_dict[
#           'detection_classes'][0].astype(np.uint8)
#       output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
#       output_dict['detection_scores'] = output_dict['detection_scores'][0]
#       if 'detection_masks' in output_dict:
#         output_dict['detection_masks'] = output_dict['detection_masks'][0]
#   return output_dict['detection_boxes'],output_dict['detection_scores'],output_dict['detection_classes']
