import ludwig
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from ludwig.api import LudwigModel
import logging

################### DOWNLOAD AND PREPROCESS DATA ###################
# Download data
image_folder = os.path.abspath('.') + '/Flicker8k_Dataset/'
if not os.path.exists(image_folder):
    image_zip = tf.keras.utils.get_file(
        'Flickr8k_Dataset.zip',
        cache_subdir=os.path.abspath('.'),
        origin = 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip',
        extract = True
    )
    text_zip = tf.keras.utils.get_file(
        'Flickr8k_text.zip',
        cache_subdir=os.path.abspath('.'),
        origin = 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip',
        extract = True
    )
    os.remove(image_zip) # remove the zip files are extracting the data
    os.remove(text_zip)

# Use the provided splits to sort each image into training, validation, and test sets
with open("Flickr_8k.trainImages.txt",'r') as f:
    training_images = set(f.read().split('\n'))
with open("Flickr_8k.devImages.txt",'r') as f:
    validation_images = set(f.read().split('\n'))
with open("Flickr_8k.testImages.txt",'r') as f:
    test_images = set(f.read().split('\n'))

training_set = {}
validation_set = {}
test_set = {}
with open("Flickr8k.token.txt",'r') as f:
    line = f.readline().split('#')
    while(len(line) >= 2):
        if(line[1][0] != '0'):                  # only use the first caption
            line = f.readline().split('#')
            continue
        line[1] = line[1].strip("\n01234.\t ")  # strip off special characters
        if line[0] in training_images:
            training_set[image_folder + line[0]] = line[1]
        elif line[0] in validation_images:
            validation_set[image_folder + line[0]] = line[1]
        elif line[0] in test_images:
            test_set[image_folder + line[0]] = line[1]
        else:
            pass
        line = f.readline().split('#')

# convert data to dataframes
training_df = pd.DataFrame(list(training_set.items()), columns=['image_path', 'caption'])
validation_df = pd.DataFrame(list(validation_set.items()), columns=['image_path', 'caption'])
test_df = pd.DataFrame(list(test_set.items()), columns=['image_path', 'caption'])

################### TRAINING ###################
# Define configuration
config = {
    "input_features": [
        {
            "name": "image_path",
            "type": "image",
            "encoder": "stacked_cnn",
            "preprocessing": {      # resize every image to be the same size
                "height": 300,
                "width": 300,
                "resize_method": "interpolate"
            }
        }
    ],
    "output_features": [
        {            
            "name": "caption",
            "type": "text",
            "level": "word",
            "decoder": "generator",
            "cell_type": "lstm"
        }
    ]
}

# Initialize and train a LudwigModel
model = LudwigModel(config, logging_level=logging.INFO)
train_stats, _, _ = model.train(
    training_set=training_df,
    validation_set=validation_df,
    test_set=test_df,
    experiment_name='image_captioning',
    model_name='example',
)

################### VISUALIZATION ###################
# Generate learning curves
from ludwig.visualize import learning_curves
learning_curves(
    train_stats_per_model = train_stats, 
    output_feature_name = "caption",
    output_directory='./visualizations',
    file_format='png'
)

################### PREDICTIONS ###################
# Predict over the test set
predictions, _ = model.predict(dataset=test_df)
predictions_df = test_df.assign(predictions = predictions["caption_predictions"])

# Print out a sample prediction
from PIL import Image
print(predictions_df.iloc[0]["predictions"])
Image.open(predictions_df.iloc[0]["image_path"])