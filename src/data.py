from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import glob

import numpy as np
import os
import shutil
from random import shuffle
from scipy.misc import imread, imsave
from sklearn.feature_extraction.image import extract_patches


def database_balancing(images_path='data', data_path='output', score_train=0.75, score_validation=0.15):
    """
    Random database balancing weighted by score_train and score_validation.
    """

    assert os.path.exists(images_path)
    img_path = os.path.join(images_path, "img")
    lab_path = os.path.join(images_path, "labels")

    train_path = os.path.join(data_path, "balance", "train")
    validation_path = os.path.join(data_path, "balance", "validation")
    test_path = os.path.join(data_path, "balance", "test")
    labels_path = os.path.join(data_path, "balance", "labels")

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(validation_path):
        os.makedirs(validation_path)

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    if not os.path.exists(labels_path):
        os.makedirs(labels_path)

    img_list = glob.glob(img_path + '/*')
    labels_list = glob.glob(lab_path + '/*')
    shuffle(img_list)

    num_img = len(img_list)
    num_train = int(np.ceil(num_img * score_train))
    num_validation = int(np.ceil(num_img * score_validation))

    train_list = img_list[0:num_train]
    validation_list = img_list[num_train:num_train + num_validation]
    test_list = img_list[num_train + num_validation:num_img]

    for name in zip(train_list):
        shutil.copy(name[0], train_path)

    for name in zip(validation_list):
        shutil.copy(name[0], validation_path)

    for name in zip(test_list):
        shutil.copy(name[0], test_path)

    for name in zip(labels_list):
        shutil.copy(name[0], labels_path)


def create_patches(patch_size, patch_overlap, mode, data_path='data'):
    """
    Create patches from data_path + mode, with patch_size x patch_size and an overlapping of patch_overlap
    """

    source_img_path = os.path.join(data_path, "balance", mode)
    source_labels_path = os.path.join(data_path, "balance", "labels")

    img_path = os.path.join(data_path, "patches", mode)
    labels_path = os.path.join(data_path, "patches", "labels")

    if not os.path.exists(img_path):
        os.makedirs(img_path)

    if not os.path.exists(labels_path):
        os.makedirs(labels_path)

    filenames = os.listdir(source_img_path)
    files_list = []

    [files_list.append(os.path.splitext(name)) for name in filenames]

    for name in files_list:

        print(name[0])

        image = imread(os.path.join(source_img_path, name[0]) + name[1])
        label = imread(os.path.join(source_labels_path, name[0]) + '.png')

        image_patches = extract_patches(image, (patch_size, patch_size, 3), patch_overlap)
        label_patches = extract_patches(label, (patch_size, patch_size), patch_overlap)

        sh = image_patches.shape
        for index, item in enumerate(image_patches.reshape([sh[0] * sh[1], sh[3], sh[4], sh[5]])):
            imsave(os.path.join(img_path, name[0] + '_' + str(index) + '.jpg'), item)

        sh = label_patches.shape
        for index, item in enumerate(label_patches.reshape([sh[0] * sh[1], sh[2], sh[3]])):
            imsave(os.path.join(labels_path, name[0] + '_' + str(index) + '.png'), item)


def compute_patch_statistics(data_path, patch_size):
    """
    Compute the percentage of parasitic pixels (classes 5, 6, 7) for each training patch.
    """
    img_path = os.path.join(data_path, 'patches', 'train')
    labels_path = os.path.join(data_path, 'patches', 'labels')

    parasite_classes = [5, 6, 7]

    filenames = os.listdir(img_path)
    files_list = []
    [files_list.append(os.path.splitext(name)) for name in filenames]

    nPixels = patch_size * patch_size

    parasite_score = np.zeros(len(files_list))

    for id, name in enumerate(files_list):
        label = imread(os.path.join(labels_path, name[0]) + '.png')

        for c in parasite_classes:
            parasite_score[id] = parasite_score[id] + len(np.where(label == c)[0]) / nPixels

    np.save(os.path.join(data_path, 'parasite_score.npy'), parasite_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data operations.")
    parser.add_argument('--img_path', type=str, default=None, help='Source raw image path')
    parser.add_argument('--data_path', type=str, default=None, help='Generated data path')
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=224)
    parser.add_argument('--patch_overlap', type=int, default=112)
    parser.add_argument('--balance', dest='balance', action='store_true', help='Flag to balance the database.')
    parser.add_argument('--patches', dest='patches', action='store_true', help='Flag to create training patches.')
    parser.add_argument('--parasite_score', dest='score', action='store_true', help='Flag to create parasite_score.')

    args = parser.parse_args()

    if args.balance:

        # database balancing
        database_balancing(args.img_path, args.data_path, score_train=0.75, score_validation=0.15)

    elif args.patches:

        # train patches
        create_patches(args.patch_size, args.patch_overlap, 'train', args.data_path)

        # validation patches
        create_patches(args.patch_size, args.patch_overlap, 'validation', args.data_path)

        # test patches
        create_patches(args.patch_size, args.patch_overlap, 'test', args.data_path)

    elif args.score:

        # compute the parasitic content of the training patches
        compute_patch_statistics(args.data_path, args.patch_size)

    else:
        print("Incorrect option")
