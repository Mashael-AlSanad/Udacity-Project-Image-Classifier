import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import argparse
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import numpy as np
import json
import matplotlib.pyplot as plt
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


# Create the process_image function
def process_image(image):
    image_size = 224
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image,(image_size,image_size))
    image /= 255
    image = tf.cast(image,tf.float32)
    return image.numpy()

#Create the predict function
def predict(image_path, model, top_k):
    img = Image.open(image_path)
    image = np.asarray(img)
    modify_image = process_image(image)
    expanded_image = np.expand_dims(modify_image, axis=0)
    
    predicted = model.predict(expanded_image)
    probs = - np.partition(-predicted[0], top_k)[:top_k]
    classes = np.argpartition(-predicted[0], top_k)[:top_k]
    return probs, list(classes)


# Define arg parser
parser = argparse.ArgumentParser(description='Image Classifier Application')

parser.add_argument('image_path', action="store")
parser.add_argument('saved_model', action="store")
parser.add_argument('--category_names', action="store", dest="category_names")
parser.add_argument('--top_k', action="store", dest="top_k", type=int)


print( parser.parse_args())
args  = parser.parse_args()

image_path = args .image_path
saved_model = args .saved_model
category_names = args .category_names
top_k = args .top_k

if top_k == None:
    top_k = 5

# load model   
model = load_model(saved_model, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)

# predict image
image = np.asarray(Image.open(image_path))
probs, classes = predict(image_path, model, top_k)


if category_names !=None:
    with open(category_names, 'r') as f:
        class_names = json.load(f)
    names = [str(x+1) for x in classes]
    classes = [class_names.get(name) for name in names]

# print ot the results (top K, class name ,probabilities)
print('\n--------------------------------------------------------------')
print('\n the {} top classes:'.format(top_k))
for i in range(top_k):
    print('\n\u2022 Class: {}'.format(classes[i]), '\n\u2022 Probability: {:.4%}'.format(probs[i]))