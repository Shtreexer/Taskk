import tensorflow as tf
from tensorflow.keras.models import load_model


import numpy as np




model = load_model('models/clf/checkpoint.model.keras')


def make_pred(path = None):

  class_list = ['BEETLE', 'BUTERFLY', 'CAT', 'COW', 'DOG', 'ELEPHANT', 'GORILLA', 'HIPPO', 
              'LIARD', 'MONKEY', 'MOUSE', 'PANDA', 'SPIDER', 'TIDER', 'ZEBRA']

  data = tf.keras.utils.load_img(path)

  input_arr = tf.keras.utils.img_to_array(data)
  input_arr = tf.image.resize(input_arr,(224, 224))
  input_arr = np.array([input_arr])

  predictions = model.predict(input_arr)
  prediction = np.array([x.argmax(axis = 0) for x in (np.exp(x)/sum(np.exp(x)) for x in predictions)])[0]

  print('Pred: ' + class_list[prediction])

  return class_list[prediction]


