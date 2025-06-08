import os
import string
import numpy as np
import pandas as pd
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tqdm import tqdm
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return '<start> ' + text + ' <end>'

def load_image(img_path, target_size=(299, 299)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def extract_features(directory):
    model = InceptionV3(weights='imagenet')
    model = Model(inputs=model.input, outputs=model.layers[-2].output)
    features = {}
    for img_name in tqdm(os.listdir(directory)):
        img_path = os.path.join(directory, img_name)
        img = load_image(img_path)
        feature = model.predict(img)
        features[img_name] = feature
    return features
