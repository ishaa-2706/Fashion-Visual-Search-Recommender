import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
import pickle
from tqdm import tqdm

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded = np.expand_dims(img_array, axis=0)
    preprocessed = preprocess_input(expanded)
    result = model.predict(preprocessed).flatten()
    return result / norm(result)

filenames = []
feature_list = []

for file in tqdm(os.listdir("images")):
    path = os.path.join("images", file)
    filenames.append(path)
    feature_list.append(extract_features(path, model))

pickle.dump(feature_list, open("embeddings.pkl", "wb"))
pickle.dump(filenames, open("filenames.pkl", "wb"))

print("Embeddings generated!")