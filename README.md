# **Traffic Sign Recognition** 

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

[image1]: ./readme_images/traffic_signs.png "Traffic Signs"
[image2]: ./readme_images/training_set_class_distributions.png "Class distribution, training set"
[image3]: ./readme_images/training_set_class_distributions.png "Class distribution, validation set"
[image4]: ./readme_images/training_valid_acc.png "Training vs validation accuracy"
[image5]: ./readme_images/right_of_way.jpg "Right-of-way at the next intersection sign"
[image6]: ./readme_images/stop.jpg "Stop sign"
[image7]: ./readme_images/30km.jpg "Speed limit 30 km/h sign"
[image8]: ./readme_images/50km.jpg "Speed limit 50 km/h sign"
[image9]: ./readme_images/70km.jpg "Speed limit 70 km/h sign"
[image10]: ./readme_images/top_5_stop_sign.png "Softmax probabilities"
---

### Writeup / README

### Data Set Summary & Exploration

#### 1. Basic summary of the data set. 

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

Examples of the traffic sign input images is provided below:

![alt text][image1]
#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the class distributions for the training and validation sets.

![alt text][image2]
![alt text][image3]

We could see that some classes are more dominant in both data sets than others. Such the network is expected (after training) to perform better then other rare examples. It could also be seen that the class distribution in both sets is somehow similar.

### Design and Test a Model Architecture

#### 1. Preprocessing

A preprocessing pipeline is applied to all three sets:

##### Grayscaling

As a first step, grayscaling is performed for all images that is due to the fact that generally different color channels have little effect on classification tasks. The action also results in the network have filters with a depth of one instead of three. Thus reducing the number of parameters needed for the network

##### Normalization

Secondly, I normalized all images to have a mean of zero and unit variance. Normalization is always an essential step in preprocessing as it results in all features (pixels in this case) being in the same range. Thus the computed gradient direction of the loss function would not be dominated by features which have large values but rather the ones that impact the classification task the most.


#### 2. Network Architecture

I chose to use the leNet architecture which is a well known solution for the traffic sign classification problem. The network consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| DROPOUT				|												|
| Convolution 5x5	    | 1x1 stride, Valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16					|
| DROPOUT				|												|
| Fully connected		| 120 hidden units								|
| RELU					|												|
| DROPOUT				|												|
| Fully connected		| 84 hidden units								|
| RELU					|												|
| DROPOUT				|												|
| Fully connected		| 43 hidden units								|
| Softmax				|												|


#### 3. Training process

##### Hyperparameters

I used random search to settle on the best parameters for my model, using the validation set as a guidance on the optimal choice. In the end I settle on the following parameters.

| Paramter        		|     Value										| 
|:---------------------:|:---------------------------------------------:| 
| Optimizer     		| Adam											|
| Learning rate    		| 0.01											|
| Batch size    		| 128											|
| Epochs         		| 40											|
| Early stopping 		| wait for 5 epochs								|
| Drop out probability	| 0.25  										|

Drop out probability and early stopping were implemented as couter-measorures against overfitting. For early stopping the training loop ends if the validation loss keeps increasing for five continuous steps beyond a small tolerence. Trained weights and biases only if there is no deterioration in the validation loss.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:(at Epoch 36)
* training set accuracy of 0.937 (calculated for each update on a given batch then averaged out at the end of the epoch)
* validation set accuracy of 0.95 (calculated at the end of each epoch on the optimized weights and biases) 
* test set accuracy of 0.941
 

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

The new test accuracy was 80.0% with high confidence for the correctly classified ones (having a probability of one for the correctly predicted class always).

An example of the top five ranked classes for one of the predicted labels is shown below.

![alt text][image10]

The 50 km/hr sign was the one that was wrongly classified the reason could be that the image is a little bit rotated and zoomed out.

### Future Improvements

* Perform data augmentation on the input data, offline augmentation to be perform as means to balance the dataset.
* Add noise to some samples of the input data, inject some randomness in the data for the model to focus on the most important features needed to classify the sign thus, increasing it's ability to generalize to new examples.
* Add class weights in the loss function in proportion to the class distribution in the training set as means to weigh rare examples equally with the dominant ones across the dataset.







