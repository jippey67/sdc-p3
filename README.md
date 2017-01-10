# behavioral cloning of a car steering wheel

This repository contains the work I did for the behavioral cloning project in the Udacity Self-Driving Car Nanodegree program.

The objective of the project is to devise a model for steering a simulator car around a track, by means of end-to-end learning. This is done by measuring human behaviour while driving the simulator car around. The data provided are:
1) camera images of front looking cameras on the left, center and right of the car and 
2) steering angles used by the human driver to drive the car.
Based on these data a convolutional neural network is trained to simulate the human behaviour in order to be able to drive the car and keep the car on track. Actual driving is done based on only the images of the center camera. 
In the following sections I will discuss the model, the data and the training of the network.

## MODEL

A lot of suggestions are available what type of model to choose., both by colleague students in the udacity Slack Channels and on the internet. An interesting model was provided in the course with a link to a paper by NVIDIA. Many suggestions have in common that the used model is quite extensive in the number of variables used. This struck me as a little odd as the task at hand is 'only' to find an appropriate steering angle, based on the curves read from a camera image. I therefore tried to achieve an acceptable solution with substantially less parameters than e.g. the NVIDIA solution. The model uses 16,039 parameters.The line of thought for the model goes as follows:- In a couple of convolutional layers try to find whether a right or a left turn is coming up. Start with a relatively deep layer to make possible the recognition of various details belonging to right or left 'corners' (I started with a layer of 32 levels deep, but reduced it later to 16 without degrading performance of the system). With each level DECREASE (in contrast to what is customary in building a deep convolutiuonal net) the depth of the convolutional layer, in order to force the network (by training) to recognize the somewhat abstract concept of a right and a left turn. The final convolutional layer therefore has a depth of 2.

The convolutional layers use a filter size of 3x3. This seems appropriate as I resized then images used to a relatively small square of 64x64 pixels per layer. The first 3 layers use a stride of 2, the next 2 layers a stride of 1 as the size of the images becomes smaller and smaller with every level. Attached to the convolutional layers are a couple of fully connected layers whose task it is to calculate the optimal steering angle based on the information found in the convolutional layers. All levels (except the last one, that has to come up with both positive and negative steering angles) use a rectified linear unit as activation function. The first fully connected layer uses a dropout of 0.25 to decrease the risk of overfitting.

The question arises how to best train this model. The data provided are images with an associated steering angle. It is not certain that a provided steering angle is also the most optimal one. It could even be plain wrong when the car is accidentally steered in the wrong direction on some images, i.e. not towards the center of the road. For practical purposes I chose the mean square error (mse) as the loss function and used that with an Adam Optimizer to train the model. We will later see that the mse brings some peculiarities to the results of training. The model was built in Tensorflow using the Keras library to define the levels.

A detailed overview of the model:

____________________________________________________________________________________________________Layer (type)                     Output Shape          Param       Connected to                     ====================================================================================================convolution2d_46 (Convolution2D) (None, 31, 31, 16)    448         convolution2d_input_10[0][0]     ____________________________________________________________________________________________________convolution2d_47 (Convolution2D) (None, 15, 15, 16)    2320        convolution2d_46[0][0]           ____________________________________________________________________________________________________convolution2d_48 (Convolution2D) (None, 7, 7, 8)       1160        convolution2d_47[0][0]           ____________________________________________________________________________________________________convolution2d_49 (Convolution2D) (None, 5, 5, 4)       292         convolution2d_48[0][0]           ____________________________________________________________________________________________________convolution2d_50 (Convolution2D) (None, 3, 3, 2)       74          convolution2d_49[0][0]           ____________________________________________________________________________________________________flatten_10 (Flatten)             (None, 18)            0           convolution2d_50[0][0]           ____________________________________________________________________________________________________dropout_10 (Dropout)             (None, 18)            0           flatten_10[0][0]                 ____________________________________________________________________________________________________dense_37 (Dense)                 (None, 128)           2432        dropout_10[0][0]                 ____________________________________________________________________________________________________dense_38 (Dense)                 (None, 64)            8256        dense_37[0][0]                   ____________________________________________________________________________________________________dense_39 (Dense)                 (None, 16)            1040        dense_38[0][0]                   ____________________________________________________________________________________________________dense_40 (Dense)                 (None, 1)             17          dense_39[0][0]                   ====================================================================================================Total params: 16,039Trainable params: 16,039Non-trainable params: 0



![rgb_img](https://cloud.githubusercontent.com/assets/23193240/21798243/9cd3f472-d713-11e6-9b41-97cff3f525be.jpg)

![yuv_img](https://cloud.githubusercontent.com/assets/23193240/21798317/f66e5f40-d713-11e6-98a7-15b0c915fbe4.jpg)

![hsv_img](https://cloud.githubusercontent.com/assets/23193240/21798329/06748dce-d714-11e6-92cc-c80257bfb9bd.jpg)

![hls_img](https://cloud.githubusercontent.com/assets/23193240/21798330/095fd304-d714-11e6-8ac7-1384a6a42b1f.jpg)

![clipped_img](https://cloud.githubusercontent.com/assets/23193240/21798332/0c5dd650-d714-11e6-9821-0739ea792763.jpg)

![clip64x64_img](https://cloud.githubusercontent.com/assets/23193240/21798333/0ed6c324-d714-11e6-8272-68499a4e3132.jpg)




DATA

After a couple of unsuccesful trials to generate my own data (driving the car on my Apple Mac was in itself already difficult), I decided to make use of the data provided by Udacity. The objective of the model is to generate useful steering angles based on the images provided by the center camera, but the data also provided images of a left and right camera. However only one steering angle for all three images was provided. I added/subtracted a 'steer_correct' to left and right images to compensate for the view. Also 54% of the steering angles are 0, and 74% are smaller than 0.1 where 1.0 is the maximum steering angle. To prevent the model from generating outputs that are mostly zero or at least relatively small, I discarded images by means of a 'steer_keep_prob' probability where the assoicated steering_angle is smaller than 'steer_threshold' in a generator that for each training batch provides a set of images and steering angles. The generator also flips the image and steering angle with a probability of 0.5 and chooses what camera to use with a probability of 1/3 for each camera. After using all entries the set of images (and associated steering angles) is reshuffled to increase arbitrainess in the slection of images in batches.

Before providing the data to the generator the data was shuffled and split into a training set (90% and a validation set (10%). No test set was used as the test was done on the simulator in autonomous mode. I wrote two different generators for generating the training set and the validation set, as I only used the center_image in the validation set to calculate the mse. Also no images were discarded from the validation set based on the steering angle being smaller than 'steer_threshold'

The data were normalized in the generators to values in the range [-0.5, 0.5] in order to prevent numerical issues from happening.

To keep up with the idea of creating a as simple as possible model, I started with feeding converted to gray images to the model. This worked to the extent that the car was able to drive a couple of turns in autonomous mode, but then left the road. Increasing the correction of steering angle for the left and right camera images didn't help and also changing back to the original colorspace (RGB) didn't work out. I took an image of the spot where the car left the track and converted that to YUV, HSV and HLS color spaces. The HSV colorspace provided the best road boundary from a human visual perspective. I decided to use this colorspace for the model.

The images as provided by the camera cover the road, but also on the top a lot of air, hills and trees, and on the bottom a piece of the car's hood, that most likely don't contain information about where the road is going. I therefore decide to clip off the top 60 and the bottom 20 pixels. I also resized this 80x320 image to a square of 64x64 one to be flexible in the use of convolution filters.


TRAINING RESULTS



 



Epoch 1/20
8000/8036 [============================>.] - ETA: 0s - loss: 0.0968
/home/carnd/anaconda3/envs/carnd-term1/lib/python3.5/site-packages/keras/engine/training.py:1527: UserWarning: Epoch comprised more than `samples_per_epoch` samples, which might affect learning results. Set `samples_per_epoch` correctly to avoid this warning.
  warnings.warn('Epoch comprised more than '
8064/8036 [==============================] - 30s - loss: 0.0969 - val_loss: 0.0134
Epoch 2/20
8064/8036 [==============================] - 28s - loss: 0.0729 - val_loss: 0.0252
Epoch 3/20
8064/8036 [==============================] - 28s - loss: 0.0602 - val_loss: 0.0263
Epoch 4/20
8064/8036 [==============================] - 28s - loss: 0.0524 - val_loss: 0.0279
Epoch 5/20
8064/8036 [==============================] - 28s - loss: 0.0494 - val_loss: 0.0256
Epoch 6/20
8064/8036 [==============================] - 28s - loss: 0.0437 - val_loss: 0.0294
Epoch 7/20
8064/8036 [==============================] - 28s - loss: 0.0415 - val_loss: 0.0234
Epoch 8/20
8064/8036 [==============================] - 28s - loss: 0.0389 - val_loss: 0.0325
Epoch 9/20
8064/8036 [==============================] - 28s - loss: 0.0388 - val_loss: 0.0267
Epoch 10/20
8064/8036 [==============================] - 28s - loss: 0.0375 - val_loss: 0.0280
Epoch 11/20
8064/8036 [==============================] - 28s - loss: 0.0369 - val_loss: 0.0259
Epoch 12/20
8064/8036 [==============================] - 28s - loss: 0.0357 - val_loss: 0.0250
Epoch 13/20
8064/8036 [==============================] - 28s - loss: 0.0354 - val_loss: 0.0308
Epoch 14/20
8064/8036 [==============================] - 28s - loss: 0.0353 - val_loss: 0.0303
Epoch 15/20
8064/8036 [==============================] - 28s - loss: 0.0328 - val_loss: 0.0292
Epoch 16/20
8064/8036 [==============================] - 28s - loss: 0.0348 - val_loss: 0.0339
Epoch 17/20
8064/8036 [==============================] - 28s - loss: 0.0332 - val_loss: 0.0328
Epoch 18/20
8064/8036 [==============================] - 28s - loss: 0.0346 - val_loss: 0.0337
Epoch 19/20
8064/8036 [==============================] - 28s - loss: 0.0334 - val_loss: 0.0309
Epoch 20/20
8064/8036 [==============================] - 28s - loss: 0.0336 - val_loss: 0.0339
In [ ]:
