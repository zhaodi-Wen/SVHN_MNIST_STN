from keras.layers import Input
from keras.models import Model
from keras.layers import Activation
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout

from utils import get_initial_weights
from layers import BilinearInterpolation
from keras.utils.generic_utils import CustomObjectScope

##四个模型对应着stn和没有stn
def STN_SVHN(input_shape=(32, 32, 3), sampling_size=(28, 28), num_classes=11):
    image = Input(shape=input_shape)
    locnet = MaxPool2D(pool_size=(2, 2))(image)
    locnet = Conv2D(20, (5, 5))(locnet)
    locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    locnet = Conv2D(20, (5, 5))(locnet)
    locnet = Flatten()(locnet)
    locnet = Dense(50)(locnet)
    locnet = Activation('relu')(locnet)
    weights = get_initial_weights(50)
    locnet = Dense(6, weights=weights)(locnet)
    x = BilinearInterpolation(sampling_size)([image, locnet])
    # conv layer 1

    model = BatchNormalization()(x)
    model = Conv2D(32, (7, 7), activation='relu', padding='same')(model)
    model = MaxPool2D(pool_size=(2, 2))(model)

    # conv layer 2
    model = BatchNormalization()(model)
    model = Conv2D(64, (5, 5), activation='relu', padding='valid')(model)
    model = MaxPool2D(pool_size=(2, 2))(model)

    # conv layer 3
    model = BatchNormalization()(model)
    model = Conv2D(256, (3, 3), activation='relu', padding='valid')(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Dropout(0.5)(model)

    # fully connected layer
    model = Flatten()(model)
    model = Dense(1024, activation='relu')(model)
    model = Dense(512, activation='relu')(model)

    x1 = Dense(4, activation='softmax',name='l1')(model)
    x2 = Dense(11, activation='softmax',name='l2')(model)
    x3 = Dense(11, activation='softmax',name='l3')(model)
    x4 = Dense(11, activation='softmax',name='l4')(model)
    # x5 = Dense(11, activation='softmax',name='l5')(model)
    # x6 = Dense(11, activation='softmax',name='l6')(model)

    x = [x1, x2, x3, x4]

    model = Model(inputs=image, outputs=x)
    return model

def SVHN(input_shape=(32, 32, 3), sampling_size=(28, 28), num_classes=11):
    image = Input(shape=input_shape)
    # conv layer 1
    model = BatchNormalization()(image)
    model = Conv2D(32, (7, 7), activation='relu', padding='same')(model)
    model = MaxPool2D(pool_size=(2, 2))(model)

    # conv layer 2
    model = BatchNormalization()(model)
    model = Conv2D(64, (5, 5), activation='relu', padding='valid')(model)
    model = MaxPool2D(pool_size=(2, 2))(model)

    # conv layer 3
    model = BatchNormalization()(model)
    model = Conv2D(256, (3, 3), activation='relu', padding='valid')(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Dropout(0.5)(model)

    # fully connected layer
    model = Flatten()(model)
    model = Dense(1024, activation='relu')(model)
    model = Dense(512, activation='relu')(model)

    x1 = Dense(4, activation='softmax', name='l1')(model)
    x2 = Dense(11, activation='softmax', name='l2')(model)
    x3 = Dense(11, activation='softmax', name='l3')(model)
    x4 = Dense(11, activation='softmax', name='l4')(model)
    # x5 = Dense(11, activation='softmax',name='l5')(model)
    # x6 = Dense(11, activation='softmax',name='l6')(model)

    x = [x1, x2, x3, x4]

    model = Model(inputs=image, outputs=x)
    return model

def STN_mnist(input_shape=(60, 60, 1), sampling_size=(30, 30), num_classes=10):
    image = Input(shape=input_shape)
    locnet = MaxPool2D(pool_size=(2, 2))(image)
    locnet = Conv2D(20, (5, 5))(locnet)
    locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    locnet = Conv2D(20, (5, 5))(locnet)
    locnet = Flatten()(locnet)
    locnet = Dense(50)(locnet)
    locnet = Activation('relu')(locnet)
    weights = get_initial_weights(50)
    locnet = Dense(6, weights=weights)(locnet)
    with CustomObjectScope({'BilinearInterpolation': BilinearInterpolation}):
        # keras.models.load_models('model.h5')
        x = BilinearInterpolation(sampling_size,name='BilinearInterpolation')([image, locnet])
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dense(num_classes)(x)
    x = Activation('softmax')(x)
    return Model(inputs=image, outputs=x)


def mnist(input_shape=(60, 60, 1), sampling_size=(30, 30), num_classes=10):
    image = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), padding='same')(image)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dense(num_classes)(x)
    x = Activation('softmax')(x)
    return Model(inputs=image, outputs=x)