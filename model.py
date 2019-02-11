import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np

lines = []
#collecting the csv data from three different files :
with open('my_data/self/New/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
with open('my_data/self/test/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
with open('my_data/recovering/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images=[]
measures=[]

#collecting the images from the containing folder
for line in lines:
    source_path=line[0]
    filename = source_path.split('/')[-1]
    current_path = 'my_data/self/New/IMG/' + filename
    image = plt.imread(current_path)
    images.append(image)
    measure = float(line[3]) #steering measurements
    measures.append(measure)

    
    
#data augmentation to increase the data set
aug_imgs, aug_measures =[],[]
for img, measure in zip (images, measures):
    aug_imgs.append(img)
    aug_measures.append(measure)
    aug_imgs.append(cv2.flip(img,1))
    aug_measures.append(measure*-1.0)

X_train = np.array(aug_imgs)
Y_train = np.array(aug_measures)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout,Cropping2D, Lambda
from keras.layers.convolutional import Convolution2D

model = Sequential()

#Pre-Processing
model.add(Cropping2D(cropping=((70,25),(0,0)),input_shape=(160,320,3))) #removing the unwanted area
model.add(Lambda(lambda x: x/255.0-0.5)) #normalizing and mean centering the data

##The Archeticture
model.add(Convolution2D(24,5,5,subsample=(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

#Training Using adam optimizer
model.compile(loss='mse', optimizer='adam')


model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')