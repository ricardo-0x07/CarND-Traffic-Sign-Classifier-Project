#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./sample_images/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./new_images/img1.jpg "Traffic Sign 1"
[image5]: ./new_images/img2.jpg "Traffic Sign 2"
[image6]: ./new_images/img3.jpg "Traffic Sign 3"
[image7]: ./new_images/img4.jpg "Traffic Sign 4"
[image8]: ./new_images/img5.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ricardo-0x07/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 samples.
* The size of the validation set is 4410 samples.
* The size of test set is 12630 samples.
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data was distributed by plotting the labels against the total number of each type.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 raw pixel values of gray scale image  | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| batch normalization	|												|
| RELU					| Element wise activation thresholds at zero.   |
| Max pooling	      	| 2x2 stride, valid padding,  outputs 14x14x6 	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| batch normalization	|												|
| RELU					| Element wise activation thresholds at zero.   |
| Max pooling	      	| 2x2 stride, valid padding,  outputs 5x5x16 	|
| Flatten       		| Output 400        							|
| Fully connected		| Output 120          							|
| batch normalization	|												|
| RELU					| Element wise activation thresholds at zero.   |
| Fully connected		| Output 84        								|
| batch normalization	|												|
| RELU					| Element wise activation thresholds at zero.   |
| Dropout				| keep probability 0.5,to minimize over fitting.|
| Fully connected		| Output Layer: compute 43 class scores.        |
| Softmax				| Computes softmax cross entropy between logits |
|						|  and labels. Measure the probability error.	|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer with a learning rate of 0.0012, batch size of 128 for 20 epochs and attained a validation accuracy of 95.7%. I also used a 50% dropout to minimize chances of over-fitting. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.957 
* test set accuracy of 0.926

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?: The lenet architecture was first tried.
* What were some problems with the initial architecture?: Lacked the required pre-processing, didn't implement dropout to minimize chances of over-fitting and did not use batch normalization.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.: The learning rate was adjusted attain an optimum validation accuracy in the the specified epochs. The architecture was adjusted to include batch normalization, which reduces the shifting of the statistical distribution of the inputs to the next layer,  was applied before activation to aid faster convergence. This also make the neurons work in the linear region of the activation function improving learning and recognition performance. A for percent dropout was also implemented during training to minimize chances of over-fitting, the dropout was applied just before the output layer.
* Which parameters were tuned? How were they adjusted and why? The value selected for the dropout had to be tunned to ensure it didn't negatively effect accuracy. The learning rate was adjusted attain an optimum validation accuracy in the the specified epochs.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model? Dropout was selected to regularize the model during its training to ensure the model is able generalize well to new data and preform with an acceptable level of accuracy. Batch normalization was used to ensure faster convergence and learning. The image was converted to gray-scale to improve image recognition and classification performance. Histogram equalization was used improve image contrast with intention of also improving the models image recognition and classification performance.

If a well known architecture was chosen:
* What architecture was chosen? The Lenet architecture was chosen.
* Why did you believe it would be relevant to the traffic sign application? It was previously used to successfully classify the MNIST data set.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? The resulting accuracies for the training, validation and test sets indicate the model is not over-fitting or under-fitting the data.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The original images had various sizes and had to be resized. The varying distances and backgrounds at which were taken may prove problematic for the classification task.  

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                         |     Prediction	        					 | 
|:--------------------------------------:|:---------------------------------------------:| 
| Priority road      	                 | Priority road   								 | 
| Right-of way at the next intersection  | Right-of way at the next intersection 		 |
| Stop					                 | Stop											 |
| 30 km/h	      		                 | 30 km/h					 				     |
| General Caution			             | Yield      							         |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 92.6%. The difference may be due to inconsistency in quality of the images.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the images, the model is very sure of its predictions and ist was correct except it incorrectly predicted a "General caution" sign to be a Yield sign. Note these two signs are very similar expect the Yield sig has a "!" mark and the general caution sign does not. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Priority road   								| 
| 1.00     				| Right-of way at the next intersection 		|
| 1.00					| Stop											|
| 1.00	      			| 30 km/h					 				    |
| 1.00				    | Yield      							        |



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


