from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from collections import OrderedDict
from keras.utils import np_utils
from scipy.misc import imread, imsave
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate

from src.unet import Unet

color_code_dict = [
    [0.0, 0.0, 0.0],  # 0 - Black   - Background
    [1.0, 0.0, 0.0],  # 1 - Red     - Non-usable area
    [1.0, 0.5, 0.0],  # 2 - Orange  - Non-parasite
    [1.0, 1.0, 1.0],  # 3 - White   - Cytoplasm
    [1.0, 0.0, 1.0],  # 4 - Magenta - Nucleus
    [0.0, 0.0, 1.0],  # 5 - Blue    - Promastigote
    [0.0, 1.0, 0.0],  # 6 - Green   - Adhered
    [0.0, 1.0, 1.0],  # 7 - Cyan    - Amastigote
]

def list_directory(path):
    for filename in os.listdir(path):
        if 'jpg' in filename or 'png' in filename:
            print(filename)


def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true = np.reshape(y_true, -1)
    y_pred = np.reshape(y_pred, -1)
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def jacard_coef(y_true, y_pred):
    smooth = 1.0
    y_true = np.reshape(y_true, -1)
    y_pred = np.reshape(y_pred, -1)
    intersection = np.sum(y_true * y_pred)
    return (intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) - intersection + smooth)


def compute_dice_class(label, prediction):
    n_classes = prediction.shape[2]
    return np.array([dice_coef(np.transpose(label)[c], np.transpose(prediction)[c]) for c in range(n_classes)])


def compute_mean(label_path):
    filenames = []
    for filename in os.listdir(label_path):
        if 'png' in filename:
            filenames.append(filename)

    filenames = np.array(filenames)
    n_pixels = 0
    n_elements = 0

    for name in filenames:

        label_name = label_path + name

        gt = imread(label_name)
        gt = np.reshape(np_utils.to_categorical(gt, 8), gt.shape + (8,))

        connectivity = 4

        mask = (gt[:, :, 7] * 255).astype("uint8")
        output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)

        for label in range(1, output[0]):
            n_elements += 1
            n_pixels += len(np.where(output[1] == label)[0])

    return n_pixels, n_elements, n_pixels / n_elements


def overall_table(img_path, labels_path, weights):
    filenames = os.listdir(img_path)
    files_list = []
    [files_list.append(os.path.splitext(name)) for name in filenames]

    unet = Unet(n_class=8, dropout=0, batch_norm=True)

    global_precision = np.zeros(8)
    global_recall = np.zeros(8)
    global_f1_score = np.zeros(8)
    global_support = np.zeros(8)
    global_dice = np.zeros(8)
    n_pixels = 0
    n_samples = 0

    sample_dice = []

    for image_name in files_list:
        n_samples += 1

        image = imread(os.path.join(img_path, image_name[0]) + image_name[1])
        label = imread(os.path.join(labels_path, image_name[0]) + '.png')
        img_rows = image.shape[0] - image.shape[0] % 32
        img_cols = image.shape[1] - image.shape[1] % 32
        image = image[0:img_rows, 0:img_cols]
        label = label[0:img_rows, 0:img_cols]

        model = unet.get_unet(img_rows=img_rows, img_cols=img_cols)
        model.load_weights(weights)

        prediction = model.predict(np.expand_dims(image, axis=0))
        prediction = np.argmax(prediction[0], axis=2)

        (test_precision, test_recall, test_f1_score,
         test_support) = precision_recall_fscore_support(
            label.flatten(), prediction.flatten(), labels=np.arange(8))

        label = np.reshape(np_utils.to_categorical(label, 8), label.shape + (8,))
        dice_class = compute_dice_class(label,
                                        np.reshape(np_utils.to_categorical(prediction, 8), prediction.shape + (8,)))
        # dice_class[zero_idx] = 0

        sample_dice.append(np.delete(dice_class, np.array([1, 2])))

        global_precision += test_precision
        global_recall += test_recall
        global_f1_score += test_f1_score
        global_support += test_support
        global_dice += dice_class
        n_pixels += (img_rows * img_cols)

    statistics = (global_precision / n_samples, global_recall / n_samples, global_f1_score / n_samples
                  , global_support / n_samples, global_dice / n_samples, n_pixels, np.array(sample_dice))

    class_names = [
        "Background",
        "Cytoplasm",
        "Nucleus",
        "Promastigote",
        "Adhered",
        "Amastigote"
    ]

    legend = [
        "Black",
        "White",
        "Magenta",
        "Blue",
        "Green",
        "Cyan"
    ]

    table_dict = OrderedDict()

    round_factor = 4

    pixels = np.delete(statistics[3], np.array([1, 2])) / statistics[5]
    pixels[0] = pixels[0] + (1 - np.sum(pixels))

    table_dict['Class'] = list(range(len(class_names)))
    table_dict['Class Name'] = class_names
    table_dict['Legend'] = legend
    table_dict['Dice'] = np.round(np.delete(statistics[4], np.array([1, 2])), round_factor)
    table_dict['Precision'] = np.round(np.delete(statistics[0], np.array([1, 2])), round_factor)
    table_dict['Recall'] = np.round(np.delete(statistics[1], np.array([1, 2])), round_factor)
    table_dict['F1 score'] = np.round(np.delete(statistics[2], np.array([1, 2])), round_factor)
    table_dict['Pixels'] = np.round(pixels, round_factor)

    print(tabulate(table_dict, headers="keys", tablefmt="fancy_grid"))


def CC_score(gt, prediction):
    class_jacard = []
    class_regions = []

    gt = np.reshape(np_utils.to_categorical(gt, 8), gt.shape + (8,))

    for c in range(5, 8):
        gt_mask = gt[:, :, c]
        pr_mask = (prediction[:, :, c] * 255).astype("uint8")
        output = cv2.connectedComponentsWithStats(pr_mask, 4, cv2.CV_32S)
        for label in range(1, output[0]):
            gt_object = gt_mask[output[2][label][1]:output[2][label][1] + output[2][label][3],
                        output[2][label][0]: output[2][label][0] + output[2][label][2]]

            pr_object = output[1][output[2][label][1]:output[2][label][1] + output[2][label][3],
                        output[2][label][0]: output[2][label][0] + output[2][label][2]]
            pr_object[pr_object != 0] = 1

            class_jacard.append(jacard_coef(gt_object, pr_object))
            class_regions.append(c)

    return class_jacard, class_regions


def global_jacard(img_path, label_path, weights):
    filenames = os.listdir(img_path)
    files_list = []
    [files_list.append(os.path.splitext(name)) for name in filenames]

    class_jacard = []
    class_regions = []

    unet = Unet(n_class=8, dropout=0, batch_norm=True)

    for name in files_list:
        image = imread(os.path.join(img_path, name[0]) + name[1])
        label = imread(os.path.join(label_path, name[0]) + '.png')

        plt.imshow(image)
        img_rows = image.shape[0] - image.shape[0] % 32
        img_cols = image.shape[1] - image.shape[1] % 32

        image = image[0:img_rows, 0:img_cols]
        label = label[0:img_rows, 0:img_cols]

        model = unet.get_unet(img_rows=img_rows, img_cols=img_cols)
        model.load_weights(weights)

        prediction = model.predict(np.expand_dims(image, axis=0))[0]
        jc = CC_score(label, prediction)
        class_jacard = class_jacard + jc[0]
        class_regions = class_regions + jc[1]

    return np.array(class_jacard), np.array(class_regions)


def jaccard_table(img_path, label_path, weights):
    class_jaccard, class_regions = global_jacard(img_path, label_path, weights)
    jaccard_ranges = [0.25, 0.5, 0.75]

    class_nb = [5, 6, 7]
    class_names = [
        "Promastigote",
        "Adhered",
        "Amastigote"
    ]

    table_dict = OrderedDict()

    table_dict['Class'] = class_names

    for range in jaccard_ranges:
        values = []
        for nb in class_nb:
            values.append(np.count_nonzero(class_jaccard[class_regions == nb] > range) / len(
                class_jaccard[class_regions == nb]) * 100)
        table_dict['J>' + str(range)] = values

    mean = []
    var = []

    for nb in class_nb:
        mean.append(np.mean(class_jaccard[class_regions == nb]) * 100)
        var.append(np.var(class_jaccard[class_regions == nb]))

    table_dict['Mean'] = mean
    table_dict['Std.Dev'] = var

    print(tabulate(table_dict, headers="keys", tablefmt="fancy_grid"))


def draw_BB(image, prediction):
    for c in range(5, 8):
        pr_mask = (prediction[:, :, c] * 255).astype("uint8")
        output = cv2.connectedComponentsWithStats(pr_mask, 4, cv2.CV_32S)
        for label in range(1, output[0]):
            image = cv2.rectangle(image, (output[2][label][0], output[2][label][1]),
                                  (output[2][label][0] + output[2][label][2], output[2][label][1] \
                                   + output[2][label][3]), color_code_dict[2], 3)

    return image


def save_results(img_path, weights, outpath, save_predictions=True, save_regions=True):
    filenames = os.listdir(img_path)
    files_list = []
    [files_list.append(os.path.splitext(name)) for name in filenames]

    unet = Unet(n_class=8, dropout=0, batch_norm=True)

    for name in files_list:
        image = imread(os.path.join(img_path, name[0]) + name[1])
        img_rows = image.shape[0] - image.shape[0] % 32
        img_cols = image.shape[1] - image.shape[1] % 32

        image = image[0:img_rows, 0:img_cols]

        model = unet.get_unet(img_rows=img_rows, img_cols=img_cols)
        model.load_weights(weights)

        prediction = model.predict(np.expand_dims(image, axis=0))[0]

        if save_predictions:
            red_prediction = np.argmax(prediction, axis=2)
            pr_mat = np.zeros(red_prediction.shape + (3,))

            for num in range(len(color_code_dict)):
                pr_mat[red_prediction == num, :] = color_code_dict[num]

            imsave(outpath + "/predictions/" + name[0] + ".jpg", pr_mat)

        if save_regions:
            bb_image = draw_BB(image, prediction)
            imsave(outpath + "/regions/" + name[0] + ".jpg", bb_image)