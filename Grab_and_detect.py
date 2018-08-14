import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
import cv2
import time
import mss
import numpy
import scipy.misc
import sys
from PIL import Image


MODEL_NAME = 'inference_graph'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
NUM_CLASSES = 6
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
cap = cv2.VideoCapture(0)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize webcam feed
#video = cv2.VideoCapture(0)
#ret = video.set(3,1280)
#ret = video.set(4,720)

def screen_grab():
    with mss.mss() as sct:
        monitor = {'top': 100, 'left': 100, 'width': 640, 'height': 480}
        out = cv2.VideoWriter('caught_shot.avi', -1, 20.0, (640, 480))

        while 'Screen capturing':
            last_time = time.time()
            img = numpy.array(sct.grab(monitor))
            cv2.imshow('OpenCV/Numpy normal', img)
            img_np = np.array(img)
            out.write(img_np)
            #img = Image.open("test1.png")
           # scipy.misc.imsave('outfile.jpg',sct.grab(monitor))
           # f = open('test1.png', 'r+')
           # jpgdata = f.read()
           # f.close()
           # from_img(img)
            #from_img("45.JPG")
            #image = cv2.imread(img)
            from_img(img)
            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                print(cap)
                #screen(cap)
                break




def from_img(image):
    #cv2.imshow('OpenCV/Numpy normal', img)
    #image = cv2.imread(img)
   # img1 = cv2.imread(img)
    #image = img1
    #print(image.shape)
    image = image[:,:,0:3]
    image_expanded = np.expand_dims(image, axis=0)
    #print(image_expanded.shape)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.80)

    # All the results have been drawn on image. Now display the image.
    cv2.imshow('Object detector', image)

if __name__=="__main__":
    screen_grab()