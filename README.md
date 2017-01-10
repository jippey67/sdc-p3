# behavioral cloning of a car steering wheel

This repository contains the work I did for the behavioral cloning project in the Udacity Self-Driving Car Nanodegree program.

The objective of the project is to devise a model for steering a simulator car around a track, by means of end-to-end learning. This is done by measuring human behaviour while driving the simulator car around. The data provided are:
1) camera images of front looking cameras on the left, center and right of the car and 
2) steering angles used by the human driver to drive the car.
Based on these data a convolutional neural network is trained to simulate the human behaviour in order to be able to drive the car and keep the car on track. Actual driving is done based on only the images of the center camera. 
In the following sections I will discuss the model, the data and the training of the network.

## MODEL

A lot of suggestions are available what type of model to choose., both by colleague students in the udacity Slack Channels and on the internet. An interesting model was provided in the course with a link to a paper by NVIDIA. Many suggestions have in common that the used model is quite extensive in the number of variables used. This struck me as a little odd as the task at hand is 'only' to find an appropriate steering angle, based on the curves read from a camera image. I therefore tried to achieve an acceptable solution with substantially less parameters than e.g. the NVIDIA solution. My model uses 16,039 parameters. The line of thought for the model goes as follows: In a couple of convolutional layers try to find whether a right or a left turn is coming up. Start with a relatively deep layer to make possible the recognition of various details belonging to right or left 'corners' (I started with a layer of 32 levels deep, but reduced it later to 16 without degrading performance of the system). With each level DECREASE (in contrast to what is customary in building a deep convolutiuonal net) the depth of the convolutional layer, in order to force the network (by training) to recognize the somewhat abstract concept of a right and a left upcoming turn. The final convolutional layer therefore has a depth of 2.

The convolutional layers use a filter size of 3x3. This seems appropriate as I resized then images used to a relatively small square of 64x64 pixels per layer. The first 3 layers use a stride of 2, the next 2 layers a stride of 1 as the size of the images becomes smaller and smaller with every level. Attached to the convolutional layers are a couple of fully connected layers whose task it is to calculate the optimal steering angle based on the information found in the convolutional layers. All levels (except the last one, that has to come up with both positive and negative steering angles) use a rectified linear unit as activation function. The first fully connected layer uses a dropout of 0.25 to decrease the risk of overfitting.

The question arises how to best train this model. The data provided are images with an associated steering angle. It is not certain that a provided steering angle is also the most optimal one. It could even be plain wrong when the car is accidentally steered in the wrong direction on some images, i.e. not towards the center of the road. For practical purposes I chose the mean square error (mse) as the loss function and used that with an Adam Optimizer to train the model. We will later see that the mse brings some peculiarities to the results of training. The model was built in Tensorflow using the Keras library to define the levels.

A detailed overview of the model:

![layer-1](https://cloud.githubusercontent.com/assets/23193240/21811363/bcdeb338-d74f-11e6-89a7-ea337e3f9b05.jpg)

## DATA

After a couple of unsuccesful trials to generate my own data (driving the car on my Apple Mac was in itself already difficult), I decided to make use of the data provided by Udacity. The objective of the model is to generate useful steering angles based on the images provided by the center camera, but the data also provided images of a left and right camera. However only one steering angle for all three images was provided. I added/subtracted a 'steer_correct' of 0.1 to left and right images to compensate for the view. Also 54% of the steering angles are 0, and 74% are in the range (-0.1, 0.1) where 1.0 is the maximum steering angle. To prevent the model from generating outputs that are mostly zero or at least relatively small, I discarded images by means of a 'steer_keep_prob' probability where the assoicated steering_angle is smaller than 'steer_threshold'. 

The dataset was shuffled before I split it into a training set and a validation set (90%/10%). No test set was used as the test was done on the simulator in autonomous mode. For both the training set and the validation set, generators were created to provide batches of images and accompanying steering angles, for training and validation purposes. While the training generator randomly chose a camera (left, center or right), randomly flipped the image and randomly excluded many images with a low steering angle, the validation generator just selected center camera images without exclusions as that is the situation the car faces in autonomous mode. Also no images were discarded from the validation set based on the steering angle being smaller than 'steer_threshold'. The data were normalized in the generators to values in the range [-0.5, 0.5] in order to prevent numerical issues from happening. After using all entries in the set of images (and associated steering angles) the complete training set is reshuffled to increase arbitrariness in the selection of images in the batches and epochs.

The images as provided by the camera cover the road, but also on the top a lot of air, hills and trees, and on the bottom a piece of the car's hood. Both segments most likely don't contain information about where the road is going. I therefore decide to clip off the top 60 and the bottom 20 pixels. I also resized this 80x320 image to a square of 64x64 one to be flexible in the use of convolution filters. Below are two RGB images of the center camera, the second one without the clipped-off top and bottom.

![rgb_img](https://cloud.githubusercontent.com/assets/23193240/21798243/9cd3f472-d713-11e6-9b41-97cff3f525be.jpg)
*RGB image*

![clipped_img](https://cloud.githubusercontent.com/assets/23193240/21798332/0c5dd650-d714-11e6-9821-0739ea792763.jpg)
*clipped RGB image*

## TRAINING THE NETWORK

To keep up with the idea of creating a as simple as possible model, I started with feeding converted-to-gray images to the model. After training for a couple of epochs on a grayscale version of the images, the network was able to drive the car for quite some distance on the first track before it left the road. Changing back to the RGB colorspace didn't help as the car left the track at the same spot. I decided to analyze the spot where the difficulties arose. The RGB image above shows it on the left side. From looking at the image it is quite clear that the border and the road have a very similar color. No edges are visible. I therefore decided to experiment with colorspaces (YUV, HSV and HLS) to find one that better discerns between road and border. The following three images show the same image in these color spaces.

![yuv_img](https://cloud.githubusercontent.com/assets/23193240/21798317/f66e5f40-d713-11e6-98a7-15b0c915fbe4.jpg)
*YUV color space*

![hsv_img](https://cloud.githubusercontent.com/assets/23193240/21798329/06748dce-d714-11e6-92cc-c80257bfb9bd.jpg)
*HSV color space*

![hls_img](https://cloud.githubusercontent.com/assets/23193240/21798330/095fd304-d714-11e6-8ac7-1384a6a42b1f.jpg)
*HLS color space*

As the HSV color space best found the border of the road, I decided to continue training with images in that color space. Note that the other color spaces very clearly found the border of the road on the right side. This offers the option to provide all three colorspaces to the network (in 9 layers) in order to find the best parameters. It however turned out that solely using the HSV color space made a reliable network so I didn't pursue this option.

The next question that arises is how many epochs to train the network. Below is the output of a training run of 20 epochs. In each epoch the whole batch of the 8036 images was fed to the network in batches of 64. 

Epoch 1/20
8064/8036 [==============================] - 30s - loss: 0.0969 - val_loss: 0.0134<br>
Epoch 2/20
8064/8036 [==============================] - 28s - loss: 0.0729 - val_loss: 0.0252<br>
Epoch 3/20
8064/8036 [==============================] - 28s - loss: 0.0602 - val_loss: 0.0263<br>
Epoch 4/20
8064/8036 [==============================] - 28s - loss: 0.0524 - val_loss: 0.0279<br>
Epoch 5/20
8064/8036 [==============================] - 28s - loss: 0.0494 - val_loss: 0.0256<br>
Epoch 6/20
8064/8036 [==============================] - 28s - loss: 0.0437 - val_loss: 0.0294<br>
Epoch 7/20
8064/8036 [==============================] - 28s - loss: 0.0415 - val_loss: 0.0234<br>
Epoch 8/20
8064/8036 [==============================] - 28s - loss: 0.0389 - val_loss: 0.0325<br>
Epoch 9/20
8064/8036 [==============================] - 28s - loss: 0.0388 - val_loss: 0.0267<br>
Epoch 10/20
8064/8036 [==============================] - 28s - loss: 0.0375 - val_loss: 0.0280<br>
Epoch 11/20
8064/8036 [==============================] - 28s - loss: 0.0369 - val_loss: 0.0259<br>
Epoch 12/20
8064/8036 [==============================] - 28s - loss: 0.0357 - val_loss: 0.0250<br>
Epoch 13/20
8064/8036 [==============================] - 28s - loss: 0.0354 - val_loss: 0.0308<br>
Epoch 14/20
8064/8036 [==============================] - 28s - loss: 0.0353 - val_loss: 0.0303<br>
Epoch 15/20
8064/8036 [==============================] - 28s - loss: 0.0328 - val_loss: 0.0292<br>
Epoch 16/20
8064/8036 [==============================] - 28s - loss: 0.0348 - val_loss: 0.0339<br>
Epoch 17/20
8064/8036 [==============================] - 28s - loss: 0.0332 - val_loss: 0.0328<br>
Epoch 18/20
8064/8036 [==============================] - 28s - loss: 0.0346 - val_loss: 0.0337<br>
Epoch 19/20
8064/8036 [==============================] - 28s - loss: 0.0334 - val_loss: 0.0309<br>
Epoch 20/20
8064/8036 [==============================] - 28s - loss: 0.0336 - val_loss: 0.0339<br>

First thing that springs to mind is that the validation loss is much smaller than the training loss right from the first epoch. This is probably caused by the fact that the validation set only uses the center camera images and does not discard images with a low associated steering angle. The next remarkable thing is that the validation loss starts to increase from the very first epoch. Multiple runs were done and the minimum validation loss always occurred after the first or second epoch. As I already mentioned earlier, the use of the mean square error as the loss function has some peculiarities that we see here in action: While the car could drive around the track quite well after training two epochs, the ride was relatively rocky. Training ten epochs provided a much smoother ride, while the validation loss was higher than the minimum, which hints at overfitting. After the tenth epoch the validation loss grew a little further and the system was still able to drive the car around while rockyness increased again. I conclude that the mse is a usable indicator for training purposes, but the proof of the pudding was in the eating. This means in this case that training for around 10 epochs provides the best solution, being able to smoothly drive the car around the track. The parameters and model of training 10 epochs are available in this repository as model.h5 en model.json. The model was also able to drive the second track, being the ultimate test for the model.

