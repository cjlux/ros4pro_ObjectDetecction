"""
Object Detection From TF2 Saved Model
=====================================
"""

# %%
# This demo will take you through the steps of running an "out-of-the-box" TensorFlow 2 compatible
# detection model on a collection of images. More specifically, in this example we will be using
# the `Saved Model Format <https://www.tensorflow.org/guide/saved_model>`__ to load the model.

import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # clean tensorflow warnings:
import pathlib
import tensorflow as tf

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

## Adapted by Jean-Luc Charles (JLC) from the work of Evan Juras.
###
### JLC v1.0 2020/07/11 - add argparse to allow options in teh command line.
###                     - using PIL.Image do load image into a numpy array
###                     - using matplotlib instead of cv2 to display image.
###

# Import packages
import sys, os, argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils
matplotlib.use('TkAgg')

#
# parse command line arguments
#
parser = argparse.ArgumentParser(description="Uses a trained network to detect object in images")
parser.add_argument('-l', '--label_map_path', type=str, required=True,
                    help='path of the "label_map" file')
parser.add_argument('-s', '--saved_model_path', type=str, required=True,
                    help='path of the "saved_model" directory')
parser.add_argument('-i', '--images', type=str, required=True,
                    help='path of the image to process or the directory that contains the images to process.>')
parser.add_argument('-n', '--nb_max_object', type=int, required=False, default=4,
                    help='max number of objects to detect per image')
parser.add_argument('-t', '--threshold', type=int, required=False, default=80,
                    help='Detection theshold (percent) to display bounding boxe around detected objets.')
parser.add_argument('-v', '--verbose', action="count", help='wether to run in verbose mode or not')
args = parser.parse_args()

verbose = True if args.verbose else False

SAVED_MODEL_PATH = args.saved_model_path
LABEL_MAP_PATH   = args.label_map_path
THRESHOLD  = args.threshold/100
NB_MAX_OBJ = args.nb_max_object

if not os.path.exists(SAVED_MODEL_PATH):
    print(f"Error: "saved_model" directory <{SAVED_MODEL_PATH}> not found")
    sys.exit()
    
if not os.path.exists(LABEL_MAP_PATH):
    print(f"Error: "label_map.pbtxt" file <{LABEL_MAP_PATH}> not found")
    sys.exit()
 
if os.path.isfile(args.images):
    IMAGE_PATHS = [args.images]
elif os.path.isdir(args.images):
    IMAGE_PATHS = [os.path.join(args.images, f) for f in os.listdir(args.images) if f.lower().endswith("png") or f.lower().endswith("jpg")]
    IMAGE_PATHS.sort()

# Load the model
# ~~~~~~~~~~~~~~
# Next we load the downloaded model
import time
print('Loading model...', end='')
start_time = time.time()
# Load saved model and build the detection function
detect_fn = tf.saved_model.load(SAVED_MODEL_PATH)
end_time = time.time()
elapsed_time = end_time - start_time
print(f'Done in {elapsed_time:.2f} seconds')


# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Label maps correspond index numbers to category names, so that when our convolution network
# predicts `5`, we know that this corresponds to `airplane`.  

# Load label map data: 
category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH, use_display_name=True)

# Putting everything together
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The code shown below loads an image, runs it through the detection model and visualizes the
# detection results, including the keypoints.
#
# Note that this will take a long time (several minutes) the first time you run this code due to
# tf.function's trace-compilation --- on subsequent runs (e.g. on new images), things will be
# faster.
#
# * Print out `detections['detection_boxes']` and try to match the box locations to the boxes in the image.  Notice that coordinates are given in normalized form (i.e., in the interval [0, 1]).
# * Set ``min_score_thresh`` to other values (between 0 and 1) to allow more detections in or to filter out more detections.

#import warnings
#warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path: str):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


for image_path in IMAGE_PATHS:

    print(f'Running inference for image <{image_path}>... ', end='')

    #image_expanded = np.expand_dims(image_rgb, axis=0)
    image_np = load_image_into_numpy_array(image_path)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)

    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    if num_detections > NB_MAX_OBJ: num_detections = NB_MAX_OBJ 
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    print(detections['detection_classes'])
    print(detections['detection_scores'])
    print(detections['detection_boxes'])

    image_np_with_detections = image_np.copy()

    vis_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          line_thickness=3,
          use_normalized_coordinates=True,
          max_boxes_to_draw=num_detections,
          min_score_thresh=THRESHOLD,
          agnostic_mode=False)
    
    plt.figure()
    plt.imshow(image_np_with_detections)
    plt.show()

