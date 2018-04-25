from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from collections import OrderedDict
from keras import backend as K
from keras.utils import np_utils
from scipy.misc import imread, imsave
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate
from scipy.ndimage import binary_fill_holes, generate_binary_structure

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

unet = Unet(n_class=8, dropout=0, batch_norm=True)


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


def predict(image_name=None, label_name=None, weights=None):
    label_name = label_name.split('.')[0] + '.png'

    assert os.path.exists(image_name), 'input image not found'
    assert os.path.exists(label_name), 'input label not found'
    assert os.path.exists(weights), 'weights file not found'

    image = imread(image_name)
    label = imread(label_name)

    img_rows = image.shape[0] - image.shape[0] % 32
    img_cols = image.shape[1] - image.shape[1] % 32

    image = image[0:img_rows, 0:img_cols]
    label = label[0:img_rows, 0:img_cols]

    model = unet.get_unet(img_rows=img_rows, img_cols=img_cols)
    model.load_weights(weights)

    prediction = model.predict(np.expand_dims(image, axis=0))
    prediction = np.argmax(prediction[0], axis=2)
    pr_mat = np.zeros(prediction.shape + (3,))

    for num in range(len(color_code_dict)):
        pr_mat[prediction == num, :] = color_code_dict[num]

    fig, axes = plt.subplots(1, 2, figsize=(15, 15))

    axes[0].imshow(image)
    axes[0].title.set_text('Source image')
    axes[0].axis('off')

    axes[1].imshow(pr_mat)
    axes[1].title.set_text('Prediction')
    axes[1].axis('off')

    fig.tight_layout()

    # Evaluation
    # label = np.reshape(np_utils.to_categorical(label, 8), label.shape + (8,))
    # prediction = np.reshape(np_utils.to_categorical(prediction, 8), prediction.shape + (8,))

    (test_precision, test_recall, test_f1_score,
     test_support) = precision_recall_fscore_support(
        label.flatten(), prediction.flatten())

    zero_idx = np.where(np.array([len(np.where(label == i)[0]) for i in range(8)]) == 0)[0]

    for idx in zero_idx:
        test_precision = np.insert(test_precision, idx, 0)
        test_recall = np.insert(test_recall, idx, 0)
        test_f1_score = np.insert(test_f1_score, idx, 0)
        test_support = np.insert(test_support, idx, 0)

    class_names = [
        "Background",
        "Non-usable area",
        "Non-parasite",
        "Cytoplasm",
        "Nucleus",
        "Promastigote",
        "Adhered",
        "Amastigote"
    ]

    legend = [
        "Black",
        "Red",
        "Orange",
        "White",
        "Magenta",
        "Blue",
        "Green",
        "Cyan",
    ]

    table_dict = OrderedDict()
    label = np.reshape(np_utils.to_categorical(label, 8), label.shape + (8,))
    dice_class = compute_dice_class(label, np.reshape(np_utils.to_categorical(prediction, 8), prediction.shape + (8,)))
    dice_class[zero_idx] = 0

    round_factor = 4

    table_dict['Class'] = list(range(len(class_names)))
    table_dict['Class Name'] = class_names
    table_dict['Legend'] = legend
    table_dict['Dice'] = np.round(dice_class, round_factor)
    table_dict['Precision'] = np.round(test_precision, round_factor)
    table_dict['Recall'] = np.round(test_recall, round_factor)
    table_dict['F1 score'] = np.round(test_f1_score, round_factor)
    table_dict['Pixels'] = np.round(test_support / (img_rows * img_cols), round_factor)
    print(tabulate(table_dict, headers="keys", tablefmt="fancy_grid"))

    return test_precision, dice_class


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


def global_stadistics(img_path, labels_path, weights):
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

        print(image_name)
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

        """
        zero_idx = np.where(np.array([len(np.where(label == i)[0]) for i in range(8)]) == 0)[0]


        for idx in zero_idx:
            test_precision = np.insert(test_precision, idx, 0)
            test_recall = np.insert(test_recall, idx, 0)
            test_f1_score = np.insert(test_f1_score, idx, 0)
            test_support = np.insert(test_support, idx, 0)
        
        """

        label = np.reshape(np_utils.to_categorical(label, 8), label.shape + (8,))
        dice_class = compute_dice_class(label,
                                        np.reshape(np_utils.to_categorical(prediction, 8), prediction.shape + (8,)))
        #dice_class[zero_idx] = 0

        sample_dice.append(np.delete(dice_class, np.array([1,2])))

        global_precision += test_precision
        global_recall += test_recall
        global_f1_score += test_f1_score
        global_support += test_support
        global_dice += dice_class
        n_pixels += (img_rows * img_cols)

    return (global_precision/n_samples, global_recall/n_samples, global_f1_score/n_samples
            , global_support/n_samples, global_dice/n_samples, n_pixels, np.array(sample_dice))


def plot_table(statistics):

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

    pixels = np.delete(statistics[3], np.array([1,2])) / statistics[5]
    pixels[0] = pixels[0] + (1 - np.sum(pixels))

    table_dict['Class'] = list(range(len(class_names)))
    table_dict['Class Name'] = class_names
    table_dict['Legend'] = legend
    table_dict['Dice'] = np.round(np.delete(statistics[4], np.array([1,2])), round_factor)
    table_dict['Precision'] = np.round(np.delete(statistics[0], np.array([1,2])), round_factor)
    table_dict['Recall'] = np.round(np.delete(statistics[1], np.array([1,2])), round_factor)
    table_dict['F1 score'] = np.round(np.delete(statistics[2], np.array([1,2])), round_factor)
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
              output[2][label][0]: output[2][label][0]+ output[2][label][2]]

            pr_object = output[1][output[2][label][1]:output[2][label][1] + output[2][label][3],
              output[2][label][0]: output[2][label][0]+ output[2][label][2]]
            pr_object[pr_object!=0] = 1

            class_jacard.append(jacard_coef(gt_object, pr_object))
            class_regions.append(c)


    return class_jacard, class_regions

def global_jacard(img_path, label_path, weights):
    filenames = os.listdir(img_path)
    files_list = []
    [files_list.append(os.path.splitext(name)) for name in filenames]

    class_jacard = []
    class_regions = []

    for name in files_list:
        print(name[0])
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

def save_predictions(img_path, weights, outpath):
    filenames = os.listdir(img_path)
    files_list = []
    [files_list.append(os.path.splitext(name)) for name in filenames]



    for name in files_list:
        print(name[0])
        image = imread(os.path.join(img_path, name[0]) + name[1])
        img_rows = image.shape[0] - image.shape[0] % 32
        img_cols = image.shape[1] - image.shape[1] % 32

        image = image[0:img_rows, 0:img_cols]

        model = unet.get_unet(img_rows=img_rows, img_cols=img_cols)
        model.load_weights(weights)

        prediction = model.predict(np.expand_dims(image, axis=0))[0]
        prediction = np.argmax(prediction, axis=2)
        pr_mat = np.zeros(prediction.shape + (3,))

        for num in range(len(color_code_dict)):
            pr_mat[prediction == num, :] = color_code_dict[num]

        imsave(outpath + "/" + name[0] + ".png", pr_mat)

def save_labels(labels_path, outpath):
    filenames = os.listdir(labels_path)
    files_list = []
    [files_list.append(os.path.splitext(name)) for name in filenames]

    for name in files_list:
        print(name[0])
        image = imread(os.path.join(labels_path, name[0]) + name[1])
        img_rows = image.shape[0] - image.shape[0] % 32
        img_cols = image.shape[1] - image.shape[1] % 32

        image = image[0:img_rows, 0:img_cols]


        mat = np.zeros(image.shape + (3,))

        for num in range(len(color_code_dict)):
            mat[image == num, :] = color_code_dict[num]

        imsave(outpath + "/" + name[0] + ".png", mat)

def new_labels(old_labels_path, new_labels_path, output_path):
    filenames = os.listdir(old_labels_path)
    files_list = []
    [files_list.append(os.path.splitext(name)) for name in filenames]

    for name in files_list:
        print(name[0])
        old_label = imread(os.path.join(old_labels_path, name[0]) + name[1])
        new_label = imread(os.path.join(new_labels_path, name[0]) + name[1])

        # create new masks. delete class 2 ...
        se = generate_binary_structure(2, 1)

        output = np.zeros(old_label.shape)

        output[np.where(new_label == 5)] = 5
        # output[np.where(old_label == 5)] = 5
        output[np.where(new_label == 6)] = 6
        # output[np.where(old_label == 6)] = 6

        output[np.where(old_label == 3)] = 3
        output[np.where(old_label == 7)] = 3

        #output = binary_fill_holes(output, se)

        #output[output == False] = 0
        #output[output == True] = 3

        output[np.where(new_label == 7)] = 7
        output[np.where(old_label == 4)] = 4
        output[np.where(new_label == 1)] = 1

        #output_mat = np.zeros(output.shape + (3,))
        #for num in range(len(color_code_dict)):
        #    output_mat[output == num, :] = color_code_dict[num]

        # save masks into ouput_path
        #imsave(os.path.join(output_path, name[0]) + name[1], output_mat)
        output = output.astype(np.uint8)
        imsave(os.path.join(output_path, name[0]) + name[1], output)
