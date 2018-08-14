import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import matplotlib as plt
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'inference_graph'

CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

NUM_CLASSES = 6

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

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

#video = cv2.VideoCapture("rtsp://192.168.42.1/live")

#video = cv2.VideoCapture("rtsp://freja.hiof.no:1935/rtplive/_definst_/hessdalen03.stream")
#video = cv2.VideoCapture("http://www.fling.asia/test2.MOV")
video = cv2.VideoCapture("test_og.mp4")
#video = cv2.VideoCapture("https://www.videvo.net/videvo_files/converted/2014_07/preview/Run_5_wo_metadata_h264420_720p_UHQ.mp446798.webm")
#video = vlc.MediaPlayer("http://192.168.0.136:8080/playlist.m3u")
#video.set(CV_CAP_PROP_BUFFERSIZE, 3);
#ret = video.set(3,1280)
#ret = video.set(4,720)

data = np.zeros((1140, 2560))



while(True):

    ret, frame = video.read()
   # ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,

        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.85)

    cv2.imshow('Object detector', frame)

    cv2.waitKey(1)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
    '''
    vis_util._visualize_boxes_and_keypoints(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,"use_normalized_coordinates=True,line_thickness=8,min_score_thresh=0.85")

  
    cv2.imshow('Object detector', frame)
   '''


# Clean up
video.release()
#video.release()
cv2.destroyAllWindows()

