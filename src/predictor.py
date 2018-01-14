from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import numpy as np
from scipy.misc import imread
from src.unet import Unet
import matplotlib.pyplot as plt

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

def display_legend():

    print('0 - Black   - Background')
    print('1 - Red     - Non-usable area')
    print('2 - Orange  - Non-parasite')
    print('3 - White   - Cytoplasm')
    print('4 - Magenta - Nucleus')
    print('5 - Blue    - Promastigote')
    print('6 - Green   - Adhered')
    print('7 - Cyan    - Amastigote')

def predict(image_name, weights):

    assert os.path.exists(image_name), 'input image not found'
    assert os.path.exists(weights), 'weights file not found'

    image = imread(image_name)

    img_rows = image.shape[0] - image.shape[0] % 32
    img_cols = image.shape[1] - image.shape[1] % 32
    image = image[0:img_rows, 0:img_cols]

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

