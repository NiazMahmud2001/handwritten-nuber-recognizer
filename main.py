import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np

num_images = 97721
img_shape = (28, 28, 3)

numpy_binary_thresh_train_x = np.empty((num_images, 784), dtype=np.uint8) #(num_images, 784) ==> img size ==> (num_images,28,28)
numpy_binary_thresh_train_y = np.empty(num_images, dtype=np.int32)


numpy_binary_thresh_test_x = np.empty((10000, 784), dtype=np.uint8)
numpy_binary_thresh_test_y = np.empty(10000, dtype=np.int32)

for x in range(0,10): 
    for y in range(0, 9773): 
        path = f"D:/Machine_Learning_Python/Tensorflow/handWritten_number_recognition/dataset/{x}/{x}/{y}.png"
        
        img_main = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        b, g, r, a = cv2.split(img_main)
        
        threshold = 50
        mask = (b < threshold) & (g < threshold) & (r < threshold) & (a > threshold)
        # of you found any 'black color', even halka colourfull , then oi black color k 'white' color dea replace korbo, then "THRESH_BINARY" dea img k black-white a convert korbo .... jar karone white color gula 'white' thakba and everythong else 'black color' hoea jaba 
        img_main[mask] = [255, 255, 255, 255]
        
        img_gray = cv2.cvtColor(img_main, cv2.COLOR_BGR2GRAY)
        
        ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY) 
        # convert image in binary and we this is the main image where tensorfloe will learn
        
        if img_main is None:
            continue  
        index = (x*9772)+y 
        
        temp_arr = thresh.flatten()/255 
        # if you did not make it range between 0-1 , the aquracy would be too low , cause machine can learn better on range between 0 to 1 , instead of 0 to 255
        numpy_binary_thresh_train_x[index] = temp_arr
        numpy_binary_thresh_train_y[index]=x
        
        
for x in range(0,10): 
    for y in range(0, 1000): 
        path = f"D:/Machine_Learning_Python/Tensorflow/handWritten_number_recognition/dataset/{x}/{x}/{9773+y}.png"
        
        img_main = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        b, g, r, a = cv2.split(img_main)
        
        threshold = 50
        mask = (b < threshold) & (g < threshold) & (r < threshold) & (a > threshold)
        # of you found any 'black color', even halka colourfull , then oi black color k 'white' color dea replace korbo, then "THRESH_BINARY" dea img k black-white a convert korbo .... jar karone white color gula 'white' thakba and everythong else 'black color' hoea jaba 
        img_main[mask] = [255, 255, 255, 255]
        
        img_gray = cv2.cvtColor(img_main, cv2.COLOR_BGR2GRAY)
        
        ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY) 
        
        if img_main is None:
            continue
        index = (x*1000)+y  
          
        temp_arr = thresh.flatten()/255 
        # if you did not make it range between 0-1 , the aquracy would be too low , cause machine can learn better on range between 0 to 1 , instead of 0 to 255
        numpy_binary_thresh_test_x[index] = temp_arr
        numpy_binary_thresh_test_y[index] = x
        
# print(numpy_binary_thresh_train_x.shape)
# print(numpy_binary_thresh_train_y.shape)

# plt.matshow(numpy_binary_thresh_train_x[0]) # showing images from numpy to matplotlib 
# plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

model = Sequential()
model.add(Dense(100 , input_shape=(784,), activation="relu"))
# this 1st hidden layer having 100 neurons, 
# N:B=> hidden layer can have less then "input_shape" neurons
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation="sigmoid"))
# this is the output layer having 10 neurons , identifying 10 integer data
# if you are having hidden layer then in output layer , you dont need to define the input_shape here
# Desnse => one neuron is connected with all other neuron
# => 10 neurons and one neuron calcualtes 28*28=784 data
# activation function are: signoid, relu, softmax

model.compile(
    optimizer="adam", 
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)

model.fit(numpy_binary_thresh_train_x , numpy_binary_thresh_train_y, epochs=5)
# epochs => one neuron loop 5 time each to learn

# on training: if the acquiracy>=0.9 then .... good ... else optimize the code again



machine_predected_complex_array = model.predict(numpy_binary_thresh_test_x)
machine_predected_number = [np.argmax(i) for i in machine_predected_complex_array]
# "machine_predected_number" ==> is the the actual number that machine detected from your test data 
print(len(machine_predected_number))
print(len(numpy_binary_thresh_test_y))

wrongNumber = 0
for x in range(10000): 
    if machine_predected_number[x] != numpy_binary_thresh_test_y[x]: 
        wrongNumber +=1
        
print("Wrong estimation: ", wrongNumber)

















