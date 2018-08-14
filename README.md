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
   
   
## Understanding Tensorflow Object Detection
The Tensorflow object detection API makes decisions based on what its trained, the training data is kept in our case is kept in the images folder. Our training data should generally be split into two groups of data which is [Train and Test](https://github.com/dan62/Tensorflow-Object-Detection/tree/master/images), we do this so that the model tests the trained model using a good random mix of test data, with more itterations of training the model becomes smarter as it adjusts the Weights and Biases appropriatly based on the correctly predicted frames. It is highly recommended to split the training data 80% of images should go to train and 20% goes to the test folder, we should also ensure a random mix of images in both folders, however the test data should be relavent to what we trained it on or else our model will never achieve 100% success. 

The training data and test data contains two kinds of data which are images and xml label/dimensions files, the xml dimentions files contain
the labeling dimentions of the marked images, it is important to understand that these xml files have to be created when labeling the images, there are softwares which allow us to do this i suggest to use [LabelImg](https://github.com/tzutalin/labelImg) for Windows. This software will allow you to mark the images and label them automatically creating the xml files.


