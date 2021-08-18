# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./sign.png "Visualization"
[image2]: ./training_data.jpg "Training Data Distribution"
[image3]: ./validation_data.jpg "Validation Data Distribution"
[image4]: ./testing_data.jpg "Test Data Distribution"
[image5]: ./test_new_images/original/bumpy.jpg "Traffic Sign 1"
[image6]: ./test_new_images/original/general_caution.jpg "Traffic Sign 2"
[image7]: ./test_new_images/original/slippery.jpg "Traffic Sign 3"
[image8]: ./test_new_images/original/wild_animal.jpg "Traffic Sign 4"
[image9]: ./test_new_images/original/work.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas and numpy libraries to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed in different classes.

![alt text][image2]
![alt text][image3]
![alt text][image4]

As it is shown in the images, the data is not evenly distributed between all the classes. 

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to normalized the image data and bring all the variables to the same range. This will reduce the dependency on on the scale of the parameters.

![alt text][image1]

Generating additional data with ImageDataGenerator method from preprocessing.image module of keras would be beneficial to avoid overfitting.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers with total of 64,811 parameters to train:

Model: "sequential"
_________________________________________________________________
Layer (type)                 | Output Shape             | Param #   
=================================================================
Input			             | (None, 32, 32, 3)        | 0       
_________________________________________________________________
conv2d (Conv2D)              | (None, 28, 28, 6)        | 456       
_________________________________________________________________
max_pooling2d (MaxPooling2D) | (None, 14, 14, 6)        | 0         
_________________________________________________________________
activation (Activation)      | (None, 14, 14, 6)        | 0         
_________________________________________________________________
dropout (Dropout)            | (None, 14, 14, 6)        | 0         
_________________________________________________________________
conv2d_1 (Conv2D)            | (None, 10, 10, 16)       | 2416      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 | (None, 5, 5, 16)         | 0         
_________________________________________________________________
activation_1 (Activation)    | (None, 5, 5, 16)         | 0         
_________________________________________________________________
dropout_1 (Dropout)          | (None, 5, 5, 16)         | 0         
_________________________________________________________________
flatten (Flatten)            | (None, 400)              | 0         
_________________________________________________________________
dense (Dense)                | (None, 120)              | 48120     
_________________________________________________________________
dense_1 (Dense)              | (None, 84)               | 10164     
_________________________________________________________________
dense_2 (Dense)              | (None, 43)               | 3655      
_________________________________________________________________
Total params: 64,811
Trainable params: 64,811
Non-trainable params: 0
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, using Lenet arcitecture with adding two dropout layers after each convolutions to prevent over fitting. Padding is set to valid for both convolutional layers, and activation functions for all layers except the last one are set to relu. The last layer activation function is softmax. I, also used Adam optimizer and sparse categorical crossentropy loss operation. The hyper parameters are choosed as follow:
EPOCHS = 15
BATCH_SIZE = 128
learning_rate = 0.001
dropout rate = 0.3


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.9543
* validation set accuracy of 0.9336 
* test set accuracy of 0.9030

I started working with Tensorflow 1.0, but as it is outdated and the Tensorflow 2.0 is much more convenient to work with, I developed a model using Keras based on Lenet arcitecture. In first few attempts with learning rate of 0.01 and 10 epochs, the model did not satisfy the required 0.93 accuracy on validation set. By settig learning rate to 0.001, and incresing the number of epochs, the model showed the characteristics of being overfitted since the training dataset accuracy was much higher than the validation dataset. Therefore, to solve the overfitting problem two dropout layers with rate of 0.3 was added after each convolutional layers.
More then 90% accuracy on evaluating test dataset along with the high training and validation accuracy shows that the model is suitable classifier for German traffic signs.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

The images of bumpy and sippery road signs are difficult to predict because of the lack of data. Also, the backgrounds of the images make things worse as the images in provided training dataset are mostly without backgrounds. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        | Prediction(original)	  Prediction(cropped)	| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy Road     		| Right-of-way    		| Bumpy Road			| 
| General Caution  		| Pedestrians 			| General Caution		|
| Slippery Road			| Bicycles Crossing		| Slippery Road			|
| Wild Animals Crossing	| Bicycles Crossing		| Wild Animals Crossing	|
| Road Work				| Road Work      		| Road Work				|


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20% on original images. By cropping the images(removing the backgrounds) the accuracy went up to 100%. This compares favorably to the accuracy on the test set of more than 90%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the second to last part of the Ipython notebook.

For the second image, the model is relatively sure that this is a Pedestrians sign (probability of .85), and the image does contain a Bumpy Road sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|  .85         			| Pedestrians   								| 
|  .14     				| Right-of-wa 									|
|  .01					| Children crossing								|
|  --	      			| --					 						|
|  --				    | --      										|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


