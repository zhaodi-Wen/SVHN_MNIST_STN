# -*- coding: utf-8 -*-
'''
# @Time    : 19-6-10 下午9:29
# @Author  :  LXF && ZXP && WZD
# @FileName: train_mnist_without_stn.py
--------------------- 
'''


import keras.backend as K
from data_manager import ClutteredMNIST
from visualizer import plot_mnist_sample
from visualizer import print_evaluation
from visualizer import plot_mnist_grid
from STN import mnist
from keras.utils import plot_model
from  keras.optimizers import Adam,SGD
from keras.callbacks import TensorBoard, EarlyStopping
import os
os.environ["CUDA_DEVICE_ORDER"]='1'

dataset_path = "data/mnist_cluttered_60x60_6distortions.npz"
batch_size = 256
num_epochs = 30

data_manager = ClutteredMNIST(dataset_path)
train_data, val_data, test_data = data_manager.load()
x_train, y_train = train_data
# plot_mnist_sample(x_train[7])


model = mnist()

model.compile(loss='categorical_crossentropy', optimizer='sgd')
model.summary()
plot_model(model, to_file='mnist_without_stn.png')

tb = TensorBoard('log/mnist/',  write_graph=True, write_images=True)


callbacks = [tb]
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50, batch_size=64, callbacks=callbacks,
          shuffle=True, verbose=1, validation_data=test_data)
model.save('model/mnist/' + 'model.h5')