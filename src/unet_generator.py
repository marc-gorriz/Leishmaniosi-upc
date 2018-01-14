from __future__ import print_function

import os
from random import shuffle, randint

import numpy as np
from keras.utils import to_categorical
from scipy.misc import imread
from skimage.transform import rotate


class UNetGeneratorClass(object):
    def __init__(self, n_class=8, batch_size=32, apply_augmentation=True, sampling_score=None,
                 data_path='data', mode='train'):

        self.filenames = np.array(os.listdir(os.path.join(data_path, mode)))

        if sampling_score is not None and mode is 'train':
            parasite_score = np.load(os.path.join(data_path, 'parasite_score.npy'))
            thresh = np.max(parasite_score) * sampling_score
            self.filenames = self.filenames[np.where(parasite_score >= thresh)[0]]

        self.files_list = []
        [self.files_list.append(os.path.splitext(name)) for name in self.filenames]

        self.labels_path = os.path.join(data_path, 'labels')
        self.image_path = os.path.join(data_path, mode)

        self.batch_size = batch_size
        self.n_class = n_class

        self.apply_augmentation = apply_augmentation

    def data_augmentation(self, data):

        if self.apply_augmentation:
            h_i_mirror = np.fliplr(data)
            v_i_mirror = np.flipud(data)

            delta = randint(0, 11)

            if delta == 0:
                return data
            elif delta == 1:
                return rotate(data, angle=180, resize=False)
            elif delta == 2:
                return rotate(data, angle=90, resize=True)
            elif delta == 3:
                return rotate(rotate(data, angle=90, resize=True), angle=180, resize=False)
            elif delta == 4:
                return h_i_mirror
            elif delta == 5:
                return v_i_mirror
            elif delta == 6:
                return rotate(h_i_mirror, angle=180, resize=False)
            elif delta == 7:
                return rotate(h_i_mirror, angle=90, resize=True)
            elif delta == 8:
                return rotate(rotate(h_i_mirror, angle=90, resize=True), angle=180, resize=False)
            elif delta == 9:
                return rotate(v_i_mirror, angle=180, resize=False)
            elif delta == 10:
                return rotate(v_i_mirror, angle=90, resize=True)
            elif delta == 11:
                return rotate(rotate(v_i_mirror, angle=90, resize=True), angle=180, resize=False)
        else:
            return data

    def generate(self):
        while True:

            shuffle(self.files_list)

            for nFiles in range(len(self.files_list) // self.batch_size):
                batch_names = self.files_list[nFiles * self.batch_size:(nFiles + 1) * self.batch_size]

                img_batch = []
                lab_batch = []

                for image_name in batch_names:
                    self.augmentation = randint(0, 11)
                    image = imread(
                        os.path.join(self.image_path, image_name[0]) + image_name[1])

                    label = imread(
                        os.path.join(self.labels_path, image_name[0]) + '.png')

                    img_batch.append(self.data_augmentation(image))
                    lab_batch.append(to_categorical(self.data_augmentation(label), self.n_class).reshape(
                        label.shape + (self.n_class,)))

                yield (np.array(img_batch), np.array(lab_batch))
