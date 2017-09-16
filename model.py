import csv
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

datadir = 'data/data/'
csvfile = datadir + 'driving_log.csv'

# Read driving_log.csv 
lines = []
with open( csvfile ) as input:
    reader = csv.reader( input )
    for line in reader:
        lines.append( line )
# Each line of lines is an array with format

# Remove the first line, which contains the description of each data column
lines = lines[1:]

import sklearn
from sklearn.model_selection import train_test_split

# Split lines of driving_log.csv into training and validation samples
# 80% of the data will be used for training.
train_samples, validation_samples = train_test_split( lines, test_size=0.2 )

# Generator for fit data
def generator( samples, batch_size=32 ):
    num_samples = len( samples )

    # Loop forever, so that next() can be called on the generator
    while 1:
        sklearn.utils.shuffle( samples )

        # Loop over batches of lines read in from driving_log.csv
        for offset in range( 0, num_samples, batch_size ):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # Extract filenames
                filename_center = batch_sample[0].split('/')[-1]
                filename_left = batch_sample[1].split('/')[-1]
                filename_right = batch_sample[2].split('/')[-1]

                path_center = 'data/data/IMG/' + filename_center
                path_left = 'data/data/IMG/' + filename_left
                path_right = 'data/data/IMG/' + filename_right
                
                # Read images using mpimg.imread
                image_center = mpimg.imread( path_center )
                image_left = mpimg.imread( path_left )
                image_right = mpimg.imread( path_right )
              
                image_flipped = np.copy( np.fliplr( image_center ) )
                
                images.append( image_center )
                images.append( image_left )
                images.append( image_right )
                images.append( image_flipped )
                
                # Correction angle added (subtracted)
                correction = 0.065
                angle_center = float( batch_sample[3] )
                angle_left = angle_center + correction
                angle_right = angle_center - correction
                angle_flipped = -angle_center
                
                angles.append( angle_center )
                angles.append( angle_left ) 
                angles.append( angle_right )
                angles.append( angle_flipped )

           
            X_train = np.array( images )
            y_train = np.array( angles )
           

            yield sklearn.utils.shuffle( X_train, y_train )

print( len( train_samples ) )
print( len( validation_samples ) )

# Define generators for training and validation data, to be used with fit_generator below
train_generator = generator( train_samples, batch_size=32 )
validation_generator = generator( validation_samples, batch_size=32 )

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Crop the hood of the car and the higher parts of the images 
model.add( Cropping2D( cropping=( (50,20), (0,0) ), input_shape=(160,320,3)))
#Normalize the data.
model.add( Lambda( lambda x: x/255. - 0.5 ) )
# Convolution Layers
model.add( Convolution2D( 24, 5, 5, subsample=(2,2), activation = 'relu' ) )
model.add( Convolution2D( 36, 5, 5, subsample=(2,2), activation = 'relu' ) )
model.add( Convolution2D( 48, 5, 5, subsample=(2,2), activation = 'relu' ) )
model.add( Convolution2D( 64, 3, 3, subsample=(1,1), activation = 'relu' ) )
model.add( Convolution2D( 64, 3, 3, subsample=(1,1), activation = 'relu' ) )
# Flatten for transition to fully connected layers.
model.add( Flatten() )
# Fully connected layers
model.add( Dense( 100 ) )
model.add(Dropout(0.5))  
model.add( Dense( 50 ) )
model.add( Dense( 10 ) )
model.add( Dense( 1 ) )

# Use mean squared error for regression, and an Adams optimizer.
model.compile( loss='mse', optimizer='adam' )

train_steps = np.ceil( len( train_samples )/32 ).astype( np.int32 )
validation_steps = np.ceil( len( validation_samples )/32 ).astype( np.int32 )


model.fit_generator( train_generator, \
    steps_per_epoch = train_steps, \
    epochs=5, \
    verbose=1, \
    callbacks=None, 
    validation_data=validation_generator, \
    validation_steps=validation_steps, \
    class_weight=None, \
    max_q_size=10, \
    workers=1, \
    pickle_safe=False, \
    initial_epoch=0 )

model.save( 'model.h5' )
