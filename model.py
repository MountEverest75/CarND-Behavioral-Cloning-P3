#Step 1: Import all essential libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Convolution2D,MaxPooling2D,Flatten,Lambda
from keras.optimizers import Adam
from keras.models import model_from_json
import json
import csv

tf.python.control_flow_ops = tf

#Step 2: Define Epochs, Learning Size, Training Size and Validation Size
#Find the number if images to train in the model
import os

# Approach:1 Training data captured from Beginner track
# data_directory = '/Users/mounteverest/Desktop/CarND/IMG'
# driving_log = '/Users/mounteverest/Desktop/CarND/driving_log.csv'

data_directory = '/Users/mounteverest/Desktop/CarND/CarNDProjects/CarND-Behavioral-Cloning-P3/data/IMG'
driving_log = '/Users/mounteverest/Desktop/CarND/CarNDProjects/CarND-Behavioral-Cloning-P3/data/driving_log.csv'


# Approach:2 Training Data Captured from Advanced Track
# Training data captured from Advanced track
# data_directory = '/Users/mounteverest/Desktop/CarND/DrivingLogs/IMG_HARD'
# driving_log = '/Users/mounteverest/Desktop/CarND/DrivingLogs/driving_log_hard.csv'

# Approach:3 Training Data Captured Sample from Udacity Github
# data_directory = '/Users/mounteverest/Desktop/CarND/CarNDProjects/CarND-Behavioral-Cloning-P3/data/IMG'
# driving_log = '/Users/mounteverest/Desktop/CarND/CarNDProjects/CarND-Behavioral-Cloning-P3/data/driving_log.csv'

#Identify the number of images to train or number of images captured while training on the simulator
image_count = len([name for name in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, name))])
print("Image Count:", image_count)

# Step 3: Define all global variables or constants - number of epochs, training size, validation size, activation function
epochs = 10
training_sample_size = len([name for name in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, name))]) * 0.70
validation_sample_size = image_count - training_sample_size
learning_rate = 0.0001
activation_relu = 'relu'
print("Training Size:",int(training_sample_size))
print("Validation Size:", int(validation_sample_size))
batch_size=200
model_weights = "model.h5"

#Step 4: Read image files to identify left, center and right images by using the header column of the driving log file.
#Note: Manually added header to the driving log CSV file generated from simulator. Simulator does not add this column
training_data = pd.read_csv(driving_log, names=None)
X_train = training_data['center']
X_left  = training_data['left']
X_right = training_data['right']
Y_train = training_data['steering']
# training_data.head()

#Step 5: Methods to generate training examples
#Ideas to fine tune image processing from: https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.dpqe94iou
def read_next_image(m,lcr,X_train,X_left,X_right,Y_train):
    # Assumed the left and center cameras are approximately 1.2 meters away from center camera mounted below rear view mirror
    # Calculate adjustments - dist meters, calculate the change in steering control
    offset=1.2
    dist=100.0
    steering = Y_train[m]
    if lcr == 0:
        image = plt.imread(X_left[m].strip())
        dsteering = -offset/dist * 360/( 2*np.pi) / 25.0
        steering += dsteering
    elif lcr == 1:
        image = plt.imread(X_train[m].strip())
    elif lcr == 2:
        image = plt.imread(X_right[m].strip())
        dsteering = offset/dist * 360/( 2*np.pi)  / 25.0
        steering += dsteering
    else:
        print ('Invalid lcr value :',lcr )
    return image,steering

def random_crop(image,steering=0.0,tx_lower=-20,tx_upper=20,ty_lower=-2,ty_upper=2,rand=True):
    # Crop subsections of the image and center
    shape = image.shape
    col_start,col_end =abs(tx_lower),shape[1]-tx_upper
    horizon=60;
    bonnet=136
    if rand:
        tx= np.random.randint(tx_lower,tx_upper+1)
        ty= np.random.randint(ty_lower,ty_upper+1)
    else:
        tx,ty=0,0
    random_crop = image[horizon+ty:bonnet+ty,col_start+tx:col_end+tx,:]
    image = cv2.resize(random_crop,(64,64),cv2.INTER_AREA)
    # the steering variable needs to be updated to counteract the shift
    if tx_lower != tx_upper:
        dsteering = -tx/(tx_upper-tx_lower)/20.0
    else:
        dsteering = 0
    steering += dsteering
    return image,steering

def random_shear(image,steering,shear_range):
    rows,cols,ch = image.shape
    dx = np.random.randint(-shear_range,shear_range+1)
    #    print('dx',dx)
    random_point = [cols/2+dx,rows/2]
    pts1 = np.float32([[0,rows],[cols,rows],[cols/2,rows/2]])
    pts2 = np.float32([[0,rows],[cols,rows],random_point])
    dsteering = dx/(rows/2) * 360/(2*np.pi*25.0) / 10.0
    M = cv2.getAffineTransform(pts1,pts2)
    image = cv2.warpAffine(image,M,(cols,rows),borderMode=1)
    steering +=dsteering
    return image,steering

def random_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = 1.0 + 0.1*(2*np.random.uniform()-1.0)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def random_flip(image,steering):
    coin=np.random.randint(0,2)
    if coin==0:
        image,steering=cv2.flip(image,1),-steering
    return image,steering

def generate_training_example(X_train,X_left,X_right,Y_train):
    m = np.random.randint(0,len(Y_train))
    lcr = np.random.randint(0,3)
    image,steering = read_next_image(m,lcr,X_train,X_left,X_right,Y_train)
    image,steering = random_shear(image,steering,shear_range=40)
    image,steering = random_crop(image,steering,tx_lower=-20,tx_upper=20,ty_lower=-2,ty_upper=2)
    image,steering = random_flip(image,steering)
    image = random_brightness(image)
    return image,steering

def generate_train_batch(X_train,X_left,X_right,Y_train,batch_size = 32):
    batch_images = np.zeros((batch_size, 64, 64, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            x,y = generate_training_example(X_train,X_left,X_right,Y_train)
            batch_images[i_batch] = x
            batch_steering[i_batch] = y
        yield batch_images, batch_steering

#Step 5: Define training data
train_generator = generate_train_batch(X_train,X_left,X_right,Y_train,batch_size)

# Step 6: Define model, Train the model and Save the model
# Source: NVIDIA paper on End to End Learning for Self driving Cars http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))
# starts with five convolutional and maxpooling layers
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())

# Next, five fully connected layers
model.add(Dense(1164))
model.add(Activation(activation_relu))

model.add(Dense(100))
model.add(Activation(activation_relu))

model.add(Dense(50))
model.add(Activation(activation_relu))

model.add(Dense(10))
model.add(Activation(activation_relu))

model.add(Dense(1))
model.summary()
model.compile(optimizer=Adam(learning_rate), loss="mse")
history = model.fit_generator(train_generator, samples_per_epoch=20000, nb_epoch=epochs, verbose=1)
model.save(model_weights)
