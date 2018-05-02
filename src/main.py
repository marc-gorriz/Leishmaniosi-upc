from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import numpy as np
import os
from keras.callbacks import TensorBoard
from scipy.misc import imread, imsave
from config.configuration import Configuration
from src.unet_generator import UNetGeneratorClass

from src.unet import Unet
from src.utils import save_results, overall_table, jaccard_table

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('-c', '--config_path', type=str, default=None, help='Configuration file')
    parser.add_argument('-a', '--action', type=str, default=None, help='train or test')
    args = parser.parse_args()

    cf = Configuration(args.config_path).load()

    if args.action is 'train':

        model = Unet(n_class=cf.num_classes, dropout=cf.dropout, batch_norm=True).get_unet()
        if cf.input_weights is not None:
            model.load_weights(cf.input_weights)

        only_parasite_generator = UNetGeneratorClass(n_class=args.num_classes, batch_size=args.batch_size,
                                                     apply_augmentation=cf.parasite_augmentation,
                                                     sampling_score=cf.parasite_score,
                                                     data_path=cf.train_data_path, mode='train')

        overall_generator = UNetGeneratorClass(n_class=args.num_classes, batch_size=args.batch_size,
                                               apply_augmentation=cf.overall_augmentation,
                                               sampling_score=cf.overall_score,
                                               data_path=cf.train_data_path, mode='train')

        validation_generator = UNetGeneratorClass(n_class=cf.num_classes, batch_size=cf.batch_size,
                                                  apply_augmentation=cf.overall_augmentation,
                                                  sampling_score=cf.overall_score,
                                                  data_path=cf.train_data_path, mode='validation')

        tensorboard = TensorBoard(log_dir=cf.train_output_path, histogram_freq=0, write_graph=cf.tB_write_graph,
                                  write_images=cf.tB_write_images)

        model.fit_generator(generator=only_parasite_generator.generate(),
                            validation_data=validation_generator.generate(),
                            validation_steps=(len(validation_generator.files_list) // cf.batch_size),
                            steps_per_epoch=(len(only_parasite_generator.files_list) // cf.batch_size),
                            epochs=cf.parasite_epochs, verbose=1, callbacks=[tensorboard])

        model.fit_generator(generator=overall_generator.generate(), validation_data=validation_generator.generate(),
                            validation_steps=(len(validation_generator.files_list) // cf.batch_size),
                            steps_per_epoch=(len(overall_generator.files_list) // cf.batch_size),
                            epochs=cf.overall_epochs, verbose=1, callbacks=[tensorboard])

        model.save_weights(os.path.join(cf.train_output_path, 'weights.hdf5'))


    elif args.action is 'test':
        weights = os.path.join(cf.train_output_path, 'weights.hdf5')
        save_results(cf.test_data_path, weights, cf.train_output_path, cf.save_predictions, cf.save_regions)

        if cf.print_overall_table:
            overall_table(cf.test_data_path, cf.test_labels_path, weights)

        if cf.print_jaccard_table:
            jaccard_table(cf.test_data_path, cf.test_labels_path, weights)
