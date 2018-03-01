
# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image2]: ./sample.png "Normal Image"
[image3]: ./flippedsample.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the Nvidia model. It consists two initial layers to crop and normalize the image data. Then, 5 convolutional layers that activate with elu and run through batch normalization. Finally, the model runs through 4 fully connected layers, including the final output layer, with the elu activation and batch normalization after each layer. The code can be found in `model.py` (line 17 - 73).
~~~
def network():
    # Nvidia Model
    model = Sequential()

    model.add(Cropping2D(cropping=((60,30), (0,0)), input_shape=(160, 160, 3)))

    model.add(Lambda(lambda x: (x / 127.5) - 1))

    # First Convolutional Layer
    model.add(Conv2D(nb_filter = 24, nb_row = 5, nb_col = 5, subsample = (2, 2)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())

    # Second Convolutional Layer
    model.add(Conv2D(nb_filter = 36, nb_row = 5, nb_col = 5, subsample = (2, 2)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())

    # Third Convolutional Layer
    model.add(Conv2D(nb_filter = 48, nb_row = 5, nb_col = 5, subsample = (2, 2)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())

    # Fourth Convolutional Layer
    model.add(Conv2D(nb_filter = 64, nb_row = 3, nb_col = 3, subsample = (1, 1)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())

    # Fifth Convolutional Layer
    model.add(Conv2D(nb_filter = 64, nb_row = 3, nb_col = 3, subsample = (1, 1)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())

    model.add(Flatten())

    # First Fully Connected Layer
    model.add(Dense(100))
    model.add(Activation('elu'))
    model.add(BatchNormalization())

    # Second Fully Connected Layer
    model.add(Dense(50))
    model.add(Activation('elu'))
    model.add(BatchNormalization())

    # Third Fully Connected Layer
    model.add(Dense(10))
    model.add(Activation('elu'))
    model.add(BatchNormalization())

    # Output
    model.add(Dense(1))

    # Choose optimizer and loss
    model.compile(loss = 'mse', optimizer= 'adam')

    return model

~~~

#### 2. Attempts to reduce overfitting in the model

The model uses batch normalization which has a slight regularization effect since it adds noises to the hidden layers. Thus, batch normalization helps us avoid overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually `model.py` (line 71). As I train the network at 2 epochs and then 5 epochs, I found that the loss keeps decreasing. So eventually I increased it to 10 and got the model to drive correctly.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, flipped center image to eliminate left turn bias.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the Nvidia model as the baseline, and then I tweaked the parameters of the network like the shape of the layers to work with my data. Specifically, the Nvidia model takes input of shape 66x200x3, but I changed it to 160x160x3.

Then I added a Keras cropping layer to focus the model on the road. To combat overfitting and help with convergence, I used batch normalization after each hidden layer. I also chose to use `elu` activation function as it is very easy to use and avoid the problems of `relu`.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The Nvidia tweaks worked well at first when I trained with a small sample of the data (about 300 samples) and 2 epochs. The car was running off track at this point.

Then I ran on the full data set at 2 epochs. And the car did really well but still ran off track at the mid way point.

I then trained with 5 epochs and 10 epochs. At 5 epochs, the car still ran off the road, but I noticed that all 5 epochs saw the accuracy increasing. Thus, I increased the number of epochs and eventually got the car to drive correctly.

#### 2. Final Model Architecture

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I used sample data from Udacity only.

To augment the data set, I also flipped images and angles in order to eliminate left-turn bias, since the sample data only runs on track 1. For example, here is an image that has then been flipped:

![alt text][image2] ![alt text][image3]

I also use the left and right camera images and apply a correction factor to help the model recover from the curb.

After the collection and augmentation process, I had 32140 number of data points. I then preprocessed this data by halving the width of the images, in order to reduce the input space.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
