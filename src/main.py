from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import numpy as np
import os
from keras.callbacks import TensorBoard
from scipy.misc import imread, imsave
from unet_generator import UNetGeneratorClass

from unet import Unet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify image patches with a U-Net")
    parser.add_argument('--train_data', type=str, default="data")
    parser.add_argument('--test_data', type=str, default="data")
    parser.add_argument('--input_path', type=str, default="input")
    parser.add_argument('--output_path', type=str, default="output")
    parser.add_argument('--test_path', type=str, default="test")
    parser.add_argument('--display_step', type=int, default=20)
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--overall_epochs', type=int, default=200)
    parser.add_argument('--parasite_epochs', type=int, default=20)
    parser.add_argument('--train', dest='do_train', action='store_true', help='Flag to train or not.')
    parser.add_argument('--test', dest='do_test', action='store_true', help='Flag to test or not.')

    args = parser.parse_args()
    only_parasite_score = 0.3
    overall_score = 0.001

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.exists(args.test_path):
        os.makedirs(args.test_path)

    if args.do_train:

        model = Unet(n_class=8, dropout=0, batch_norm=True).get_unet()
        #model.load_weights('/imatge/mgorriz/work/Leishmaniosi-Project/experiments/p30-40-100-pa-50-1/unet_train weights.hdf5')

        only_parasite_generator = UNetGeneratorClass(n_class=args.num_classes, batch_size=args.batch_size,
                                                     apply_augmentation=False, sampling_score=only_parasite_score,
                                                     data_path=args.train_data, mode='train')

        overall_generator = UNetGeneratorClass(n_class=args.num_classes, batch_size=args.batch_size,
                                               apply_augmentation=False, sampling_score=overall_score,
                                               data_path=args.train_data, mode='train')

        validation_generator = UNetGeneratorClass(n_class=args.num_classes, batch_size=args.batch_size,
                                               apply_augmentation=False, sampling_score=overall_score,
                                               data_path=args.train_data, mode='test1')

        tensorboard = TensorBoard(log_dir=args.output_path, histogram_freq=0, write_graph=True, write_images=False)

        model.fit_generator(generator=only_parasite_generator.generate(), validation_data=validation_generator.generate(),
                            validation_steps=(len(validation_generator.files_list) // args.batch_size),
                            steps_per_epoch=(len(only_parasite_generator.files_list) // args.batch_size),
                            epochs=args.parasite_epochs, verbose=1, callbacks=[tensorboard])

        model.fit_generator(generator=overall_generator.generate(), validation_data=validation_generator.generate(),
                            validation_steps=(len(validation_generator.files_list) // args.batch_size),
                            steps_per_epoch=(len(overall_generator.files_list) // args.batch_size),
                            epochs=args.overall_epochs, verbose=1, callbacks=[tensorboard])

        model.save_weights(os.path.join(args.output_path, 'weights.hdf5'))
        model.save(os.path.join(args.output_path, 'model.h5'))


    elif args.do_test:

        filenames = os.listdir(os.path.join(args.test_data, 'test1'))
        files_list = []
        [files_list.append(os.path.splitext(name)) for name in filenames]

        unet = Unet(n_class=8, dropout=0, batch_norm=True)

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

        for image_name in files_list:
            image = imread(
                os.path.join(args.test_data, 'test1', image_name[0]) + image_name[1])

            label = imread(
                os.path.join(args.test_data, 'labels', image_name[0]) + '.png')

            img_rows = image.shape[0] - image.shape[0] % 32
            img_cols = image.shape[1] - image.shape[1] % 32

            image = image[0:img_rows, 0:img_cols]
            label = label[0:img_rows, 0:img_cols]

            model = unet.get_unet(img_rows=img_rows, img_cols=img_cols)

            model.load_weights(os.path.join(args.output_path, 'weights200.hdf5'))

            prediction = model.predict(np.expand_dims(image, axis=0))
            prediction = np.argmax(prediction[0], axis=2)
            gt_mat = np.zeros(label.shape + (3,))
            pr_mat = np.zeros(prediction.shape + (3,))

            for num in range(len(color_code_dict)):
                gt_mat[label == num, :] = color_code_dict[num]
                pr_mat[prediction == num, :] = color_code_dict[num]

            imsave(os.path.join(args.test_path, image_name[0] + '_p.png'), pr_mat)
            imsave(os.path.join(args.test_path, image_name[0] + '_gt.png'), gt_mat)
            imsave(os.path.join(args.test_path, image_name[0] + '_img.png'), image)
