import numpy as np
import scipy.io as  sio

class ClutteredMNIST(object):

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def to_categorical(self, y, num_classes=None):
        y = np.array(y, dtype='int').ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes))
        categorical[np.arange(n), y] = 1
        return categorical

    def load(self):
        num_classes = 10
        data = np.load(self.dataset_path)
        x_train = data['x_train']
        x_train = x_train.reshape((x_train.shape[0], 60,60,1))
        y_train = np.argmax(data['y_train'], axis=-1)
        y_train = self.to_categorical(y_train, num_classes)
        train_data = (x_train, y_train)

        x_val = data['x_valid']
        x_val = x_val.reshape((x_val.shape[0], 60,60,1))
        y_val = np.argmax(data['y_valid'], axis=-1)
        y_val = self.to_categorical(y_val, num_classes)
        val_data = (x_val, y_val)

        x_test = data['x_test']
        x_test = x_test.reshape((x_test.shape[0], 60,60,1))
        y_test = np.argmax(data['y_test'], axis=-1)
        y_test = self.to_categorical(y_test, num_classes)
        test_data = (x_test, y_test)

        return(train_data, val_data, test_data)

class ClutteredSVHN(object):

    def __init__(self, train_path,train_mean_path,
                 test_path,test_mean_path,
                 val_path,val_mean_path):
        self.train_path = train_path
        self.train_mean_path = train_mean_path
        self.test_path = test_path
        self.test_mean_path = test_mean_path
        self.val_path = val_path
        self.val_mean_path = val_mean_path


    def to_categorical(self, y, num_classes=None):
        y = np.array(y, dtype='int').ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes))
        categorical[np.arange(n), y] = 1
        return categorical

    def load(self):
        num_classes = 11
        train_data = sio.loadmat(self.train_path)
        train_mean = np.load(self.train_mean_path)
        test_data = sio.loadmat(self.test_path)
        test_mean = np.load(self.test_mean_path)
        val_data = sio.loadmat(self.val_path)
        val_mean = np.load(self.val_mean_path)

        train_dnum = train_data['X'].shape[3]
        test_dnum = test_data['X'].shape[3]
        val_dnum = val_data['X'].shape[3]

        train_images = np.array(
            [train_data['X'][:, :, :, i] for i in range(train_dnum)])
        train_images = train_images.astype(np.float32)
        train_images = [image - train_mean for image in train_images]

        test_images = np.array(
            [test_data['X'][:, :, :, i] for i in range(test_dnum)])
        test_images = test_images.astype(np.float32)
        test_images = [image - test_mean for image in test_images]

        val_images = np.array(
            [val_data['X'][:, :, :, i] for i in range(val_dnum)])
        val_images = val_images.astype(np.float32)
        val_images = [image - val_mean for image in val_images]

        # Convert images pixels from [0, 255] -> [0.0, 1.0].
        images = np.multiply(train_images, 1.0 )
        x_train = images
        x_train = x_train.reshape((x_train.shape[0], 32,32,3))
        y_train = np.argmax(train_data['y'], axis=-1)
        y_train = self.to_categorical(y_train, num_classes)
        train_data = (x_train, y_train)


        images = np.multiply(test_images, 1.0 )
        x_test = images
        x_test = x_test.reshape((x_test.shape[0], 32,32,3))
        y_test = np.argmax(test_data['y'], axis=-1)
        y_test = self.to_categorical(y_test, num_classes)
        test_data = (x_test, y_test)

        images = np.multiply(val_images, 1.0 )
        x_val = images
        x_val = x_val.reshape((x_val.shape[0], 32,32,3))
        y_val = np.argmax(val_data['y'], axis=-1)
        y_val = self.to_categorical(y_val, num_classes)
        val_data = (x_val, y_val)

        return(train_data,test_data,val_data)
