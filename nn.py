#!/usr/bin/env python3

import argparse, pathlib, os, sys

#
# parse command line arguments
#
parser = argparse.ArgumentParser(description="Uses a trained network to detect object in images")
parser.add_argument('-l', '--label_map_path', type=str, required=True,
                    help='path of the "label_map.pbtxt" file')
parser.add_argument('-s', '--saved_model_path', type=str, required=True,
                    help='path of the "saved_model/" directory')
parser.add_argument('-n', '--nb_max_object', type=int, required=False, default=4,
                    help='max number of objects to detect per image')
parser.add_argument('-t', '--threshold', type=int, required=False, default=80,
                    help='Detection theshold (percent) to display bounding boxe around detected objets.')
parser.add_argument('-v', '--verbose', action="count", help='wether to run in verbose mode or not')
args = parser.parse_args()

verbose = True if args.verbose else False

#
# Set useful names
#
SAVED_MODEL_PATH = args.saved_model_path
LABEL_MAP_PATH   = args.label_map_path
THRESHOLD  = args.threshold/100
NB_MAX_OBJ = args.nb_max_object

if not os.path.exists(SAVED_MODEL_PATH):
    print(f'Error: "saved_model" directory <{SAVED_MODEL_PATH}> not found')
    sys.exit()
    
if not os.path.exists(LABEL_MAP_PATH):
    print(f'Error: "label_map.pbtxt" file <{LABEL_MAP_PATH}> not found')
    sys.exit()
    
#
# Import other packages
#
import cv2, os, rospy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from poppy_controllers.srv import GetImage
from cv_bridge import CvBridge

import tensorflow as tf
from object_detection.utils import label_map_util

#
# do some tensorflow low level stuff...
#
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Load saved model and build the detection function
print('Loading model...', end='')
detect_fn = tf.saved_model.load(SAVED_MODEL_PATH)
print('Done!')

# Load label map data: 
category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH, use_display_name=True)

##
## ROS stuff to complete...
##

## 1/ Wait for the ROS parameter /takeImage to become True.
## 

takeImage = rospy.get_param("/takeImage")
while not takeImage:
    rospy.sleep(1)
    takeImage = rospy.get_param("/takeImage")


## 2/ Set the ROS parameter /takeImage to False.
## 

rospy.set_param("/takeImage", False)


## 3/ Get the image from the robot camera using the /get_image ROS service
## and write it (cv2.write(...) ) as "image.png".
## 

get_image = rospy.ServiceProxy("get_image", GetImage)
response  = get_image()
bridge    = CvBridge()
image     = bridge.imgmsg_to_cv2(response.image)
cv2.imwrite(f"image.png", image)


## 4/ Run the network inference to detect cube faces.
## 

print('Running inference for image.png... ')
image_np     = np.array(Image.open("image.png"))
input_tensor = tf.convert_to_tensor(image_np)  # convert input needs to be a tensor
input_tensor = input_tensor[tf.newaxis, ...]   # the model expects an array of images: add an axis
detections   = detect_fn(input_tensor)         # make the detections of the objects

num_detections = int(detections.pop('num_detections'))
if num_detections > NB_MAX_OBJ: num_detections = NB_MAX_OBJ
detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
detections['num_detections'] = num_detections
# detection_classes should be ints:
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

## 5/ Sort the lists returned by the network to arrange data  
## from the leftmost cube to the rightmost cube, based on the abcissa of 
## the top-left corner of the bounding boxes of the cubes
## Some help here : https://numpy.org/doc/stable/reference/generated/numpy.argsort.html...
## 

x_list = detections['detection_boxes'][:,1]
indexes = np.argsort(x_list)
list_label_sorted = detections['detection_classes'][indexes]
list_boxe_sorted  = detections['detection_boxes'][indexes]
list_score_sorted = detections['detection_scores'][indexes]

if verbose:
    print(list_label_sorted)
    print(list_score_sorted)
    print(list_boxe_sorted)

## 6/ Loop in the list of labels to set the parameter /label;
## set the ROS parameter /robotReady to False and wait for 
## /robotReady to be True.
## 

count = 1
for label, score in zip(list_label_sorted, list_score_sorted):
   
    
    if score >= THRESHOLD:
        print(f"object #{count}: score {score:.3f} > {THRESHOLD} is OK")
        print(f"\tset ROS param /label to {label}") 
        rospy.set_param("/label", int(label))
       
        print("\tset ROS param /robotReady to False") 
        rospy.set_param("/robotReady", False)

        robotReady = rospy.get_param("/robotReady")
        #while not robotReady:
        #   rospy.sleep(1)
        #   robotReady = rospy.get_param("/robotReady")
    else:
        print(f"object #{count}: score {score:.3f} < {THRESHOLD} => skipping")

    count += 1

