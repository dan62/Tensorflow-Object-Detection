# Tensorflow-Object-Detection
This repository explains how to setup Tensorflow  with cpu and gpu for object detection, I have also added and modified some scripts so that you can add some cool features to your model effortlessly :-) 

## Getting Started 
1. You will first need to settup the environment for the tensorflow model to work correctly, the process differs slighly 
   between [Linux Based Systems](https://github.com/dan62/Tensorflow-Object-Detection/blob/master/Linux_Settup.md) and [Windows Based Systems](https://github.com/dan62/Tensorflow-Object-Detection/blob/master/Windows_Settup.md) , please follow the correct link that identifies your system and proceed with instructions to get started
   
2. Create a folder in your C drive and name it TensorExample

3. Clone the respositoy from the official tensorflow github respository by following this [Link](https://github.com/tensorflow/models)

4. Unzip the respository and place it in the TensorExample folder 

5. Clone this respository and unzip it to the Object Detection folder located in C://TensorExample/models/research/object detection/
   ensure that you overwrite all files if prompted

6. Navigate to the directory C://TensorExample/models/research/build/lib/object_detection/utils/learning_schedules.py file and replace lines 
   167-169 with the following code:
   
   ```
   rate_index = tf.reduce_max(tf.where(tf.greater_equal(global_step, boundaries),
   list(range(num_boundaries)),
   [0] * num_boundaries))
   ```
   
## Training your model
Since this is a clean tensorflow directory, we just need to follow the following steps to start training (Please ensure you have the required environment settup):

1. Open the training folder and edit the label map with the various classes that you would like your model to detect

2. Open the pipeline.config file in the training folder and edit line 9 with the number of classes that you added to the label map, 
then go to line 129 and change num_examples to the number of images that you plan to train your model on

3. Collect as much trainiung images as you can find and place 80% of these images to the images/train folder and 0% of these images to      images/test folder  

4. Install [LabelImg](https://github.com/tzutalin/labelImg) and open it then select the train directory and start drawing boxes around the objects that you would like the model to detect and add labels to them do the same for the test folder and save each time you have marked an image

5. Open up the Anaconda command prompt and enter the following commands:

   ```
   activate tensor
 
   set PYTHONPATH=C:\TensorExample\models;C:\TensorExample\models\research;C:\TensorExample\models\research\slim
 
   cd C:\TensorExample\models\research\
 
 protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto
 
 python setup.py build
 
 python setup.py install
 
 cd object_detection
 
 python xml_to_csv.py
 
 python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
 
 python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
 
 python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
 
   ```

   6. Leave the model to train you should see the terminal flooding with text, training really depends on the hardware specifications of your pc and wheather or not you are using a gpu. Although you can exit the terminal anytime that you feel necessary the model will automatically create a checkpoint to begin from next time.

   * If you you like to retrain an existing trained model you would need to delete all the images in the images/test and images/train folders, also delete all files in the training folder except for the following faster_rcnn, labelmap, pipeline. You would also need to delete everything in the inferance_graph folder as well as test.record and train.record files. Once all is done repeat steps 1 - 6 . 

   
## Understanding Tensorflow Object Detection
The Tensorflow object detection API makes decisions based on what its trained, the training data is kept in our case is kept in the images folder. Our training data should generally be split into two groups of data which is [Train and Test](https://github.com/dan62/Tensorflow-Object-Detection/tree/master/images), we do this so that the model tests the trained model using a good random mix of test data, with more itterations of training the model becomes smarter as it adjusts the Weights and Biases appropriatly based on the correctly predicted frames. It is highly recommended to split the training data 80% of images should go to train and 20% goes to the test folder, we should also ensure a random mix of images in both folders, however the test data should be relavent to what we trained it on or else our model will never achieve 100% success. 

The training data and test data contains two kinds of data which are images and xml label/dimensions files, the xml dimentions files contain
the labeling dimentions of the marked images, it is important to understand that these xml files have to be created when labeling the images, there are softwares which allow us to do this i suggest to use [LabelImg](https://github.com/tzutalin/labelImg) for Windows. This software will allow you to mark the images and label them automatically creating the xml files.


