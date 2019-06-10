# -*- coding: utf-8 -*-
'''
# @Time    : 19-6-10 下午9:01
# @Author  :  LXF && ZXP && WZD
# @FileName: train_mnist.py
---------------------
'''

import keras.backend as K
from data_manager import *
from visualizer import plot_mnist_sample
from visualizer import print_evaluation
from visualizer import plot_mnist_grid
from STN import STN_SVHN,SVHN
from keras.utils import plot_model
from keras.optimizers import SGD, Adam, RMSprop
import h5py
from PIL import Image
from keras.callbacks import TensorBoard, EarlyStopping
from keras.utils import np_utils

batch_size = 256
num_epochs = 30


##load  data
train_dataset = h5py.File("data/train/digitStruct.mat", "r")
test_dataset = h5py.File("data/test/digitStruct.mat", "r")

def getName(dataset, index):
    names = dataset["digitStruct"]["name"]
    return ''.join([chr(c[0]) for c in dataset[names[index][0]].value])


def bboxHelper(dataset, attr):
    if (len(attr) > 1):
        attr = [dataset[attr.value[j].item()].value[0][0] for j in range(len(attr))]
    else:
        attr = [attr.value[0][0]]

    return attr


def getBbox(dataset, index):
    item = dataset[dataset["digitStruct"]["bbox"][index].item()]

    return {
        "height": bboxHelper(dataset, item["height"]),
        "label": bboxHelper(dataset, item["label"]),
        "left": bboxHelper(dataset, item["left"]),
        "top": bboxHelper(dataset, item["top"]),
        "width": bboxHelper(dataset, item["width"]),
    }


def getWholeBox(dataset, index, im):
    bbox = getBbox(dataset, index)

    im_left = min(bbox["left"])
    im_top = min(bbox["top"])
    im_height = max(bbox["top"]) + max(bbox["height"]) - im_top
    im_width = max(bbox["left"]) + max(bbox["width"]) - im_left

    im_top = im_top - im_height * 0.05  # a bit higher
    im_left = im_left - im_width * 0.05  # a bit wider
    im_bottom = min(im.size[1], im_top + im_height * 1.05)
    im_right = min(im.size[0], im_left + im_width * 1.05)

    return {
        "label": bbox["label"],
        "left": im_left,
        "top": im_top,
        "right": im_right,
        "bottom": im_bottom
    }


# Load Train Data
train_count = train_dataset["digitStruct"]["name"].shape[0]

X_train = np.ndarray(shape=(train_count, 32, 32, 3), dtype='float32')
y = {
    0: np.zeros(train_count),
    1: np.ones(train_count) * 10,
    2: np.ones(train_count) * 10,
    3: np.ones(train_count) * 10,
    4: np.ones(train_count) * 10,
    5: np.ones(train_count) * 10
}
print(y[0].shape)
for i in range(train_count):
    im = Image.open("data/train/" + getName(train_dataset, i))
    box = getWholeBox(train_dataset, i, im)
    if len(box["label"]) > 3:
        continue
    im = im.crop((box["left"], box["top"], box["right"], box["bottom"])).resize((32, 32))

    X_train[i, :, :, :] = np.array(im.resize((32, 32)), dtype='float32')

    labels = box["label"]

    y[0][i] = len(labels)

    for j in range(0, 3):
        if j < len(labels):
            if labels[j] == 10:
                y[j + 1][i] = 10
            else:
                y[j + 1][i] = int(labels[j])
        else:
            y[j + 1][i] = 10
y_train = [
    np_utils.to_categorical(y[0]),
    np_utils.to_categorical(y[1]),
    np_utils.to_categorical(y[2]),
    np_utils.to_categorical(y[3])
]
print(y_train[0].shape)
# Load Test Data
test_count = test_dataset["digitStruct"]["name"].shape[0]

X_test = np.ndarray(shape=(test_count, 32, 32, 3), dtype='float32')
y = {
    0: np.zeros(test_count),
    1: np.ones(test_count) * 10,
    2: np.ones(test_count) * 10,
    3: np.ones(test_count) * 10,
    4: np.ones(test_count) * 10,
    5: np.ones(test_count) * 10
}

for i in range(test_count):
    im = Image.open("data/test/" + getName(test_dataset, i))
    box = getWholeBox(test_dataset, i, im)
    if len(box["label"]) > 3:
        continue
    im = im.crop((box["left"], box["top"], box["right"], box["bottom"])).resize((32, 32))

    X_test[i, :, :, :] = np.array(im.resize((32, 32)), dtype='float32')

    labels = box["label"]

    y[0][i] = len(labels)

    for j in range(0, 3):
        if j < len(labels):
            if labels[j] == 10:
                y[j + 1][i] = 10
            else:
                y[j + 1][i] = int(labels[j])
        else:
            y[j + 1][i] = 10

    # if i % 500 == 0:
    #     print(i, len(y[0]))

y_test = [
    np_utils.to_categorical(y[0]),
    np_utils.to_categorical(y[1]),
    np_utils.to_categorical(y[2]),
    np_utils.to_categorical(y[3])
]


model = STN_SVHN()
# model = General_model()
model.compile(loss='categorical_crossentropy', optimizer='sgd')
model.summary()
plot_model(model, to_file='model.png')

tb = TensorBoard('log/SVHN_stn',  write_graph=True, write_images=True)

# es = EarlyStopping(monitor='val_acc', patience=5, verbose=0)

callbacks = [tb]
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=64, callbacks=callbacks,
          shuffle=True, verbose=1, validation_data=(X_test, y_test))
model.save('model/SVHN_stn/' + 'model.h5')

