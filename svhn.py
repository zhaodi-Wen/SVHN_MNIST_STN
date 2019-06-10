
# -*- coding: utf-8 -*-
'''
# @Time    : 19-6-10 下午9:01
# @Author  :  LXF && ZXP && WZD
# @FileName: train_mnist.py
---------------------
'''

import numpy as np
from tensorflow.python.framework import dtypes


class DataSet(object):

    def __init__(self, data, mean=0, dtype=dtypes.float32, reshape=True):
        """Construct a DataSet.
        `dtype` can be either `uint8` to leave the input as `[0, 255]`,
        or `float32` to rescale into `[0, 1]`.
        """
        assert 'X' in data
        assert 'y' in data
        assert data['X'].shape[3] == data['y'].shape[0]

        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

        dnum = data['X'].shape[3]

        # Covert images
        # from [rows, columns, depth, dnum]
        # to [dnum, rows, columns, depth]
        images = np.array(
            [data['X'][:, :, :, i] for i in range(dnum)])

        # Convert labels_idx to one-hot 2d matrix
        labels_idx = np.array(
            [0 if label == 10 else label for label in data['y']], dtype=int)
        labels = np.zeros((dnum, 10))
        labels[np.arange(dnum), labels_idx] = 1

        # Minus mean
        images = images.astype(np.float32)
        images = [image - mean for image in images]

        # Convert images pixels from [0, 255] -> [0.0, 1.0].
        if dtype == dtypes.float32:
            images = np.multiply(images, 1.0 / 255.0)
        elif dtype == dtypes.uint8:
            images = images.astype(np.uint8)

        # Convert shape
        # from [num examples, rows, columns, depth]
        # to [num examples, rows*columns*depth]
        if reshape:
            images = images.reshape(
                images.shape[0],
                images.shape[1] * images.shape[2] * images.shape[3])

        self._images = images
        self._labels = labels
        self._num_examples = dnum
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    # Return the next `batch_size` examples from this data set
    # Skip the last batch while its number is not enough for batch_size
    def next_batch(self, batch_size):
        assert batch_size <= self._num_examples

        # Move the index
        start = self._index_in_epoch
        end = self._index_in_epoch + batch_size
        self._index_in_epoch += batch_size

        # Finished epoch
        if self._index_in_epoch > self._num_examples:
            # Add one epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch, reset the index
            start = 0
            end = batch_size
            self._index_in_epoch = batch_size
        return self._images[start:end], self._labels[start:end]
