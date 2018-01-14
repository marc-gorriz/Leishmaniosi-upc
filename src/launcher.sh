#!/bin/bash

TRAIN_DATA="/imatge/mgorriz/work/Leishmaniosi-Project/data/patches"
TEST_DATA="/imatge/mgorriz/work/Leishmaniosi-Project/data/balance"
TRAIN_INPUT_PATH="/imatge/mgorriz/work/Leishmaniosi-Project/train0/unet_train/"
TRAIN_OUTPUT_PATH="/imatge/mgorriz/work/Leishmaniosi-Project/train0/unet_train/"
#TEST_PATH="/imatge/mgorriz/work/Leishmaniosi-Project/train_aug/unet_test/overall150"
TEST_PATH="/imatge/mgorriz/work/Leishmaniosi-Project/train0/unet_test/"
#OUTPUT_PATH="/imatge/mgorriz/work/Leishmaniosi-Project/train0"

if [ "$1" = "train" ]
then
    srun --pty --gres=gpu:1,gmem:11G --mem 20G python3 main.py --train_data $TRAIN_DATA --output_path $TRAIN_OUTPUT_PATH --test_path $TEST_PATH --input_path $TRAIN_INPUT_PATH --train
elif [ "$1" = "test" ]
then
    srun --pty --gres=gpu:1,gmem:11G --mem 20G python3 main.py --test_data $TEST_DATA --output_path $TRAIN_OUTPUT_PATH --test_path $TEST_PATH --input_path $TRAIN_INPUT_PATH --test
else
    echo "Incorrect option"
fi