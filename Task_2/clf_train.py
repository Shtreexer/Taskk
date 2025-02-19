

from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf


def prep_datasets(path = None, valid_size = 0.2):

  H = 224
  W = 224
  BATCH_SIZE = 32
  PATH = path

  train_data = tf.keras.utils.image_dataset_from_directory(
    PATH,
    validation_split = valid_size,
    subset="training",
    seed = 9,
    image_size=(H, W),
    batch_size=BATCH_SIZE)
  
  valid_data = tf.keras.utils.image_dataset_from_directory(
    PATH,
    validation_split = valid_size,
    subset = "validation",
    seed = 9,
    image_size=(H, W),
    batch_size=BATCH_SIZE)
  

  train_data = train_data.cache().prefetch(tf.data.AUTOTUNE)
  valid_data = valid_data.cache().prefetch(tf.data.AUTOTUNE)


  return train_data, valid_data


def train_clf(train = None, valid = None, num_epochs = 10):

  sgd = tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9, nesterov = True)

  model = tf.keras.Sequential()
  model.add(tf.keras.applications.ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'))
  model.add(tf.keras.layers.Dense(15, activation = 'softmax'))
  model.layers[0].trainable = False

  model.compile(optimizer = sgd, 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
              metrics = ['accuracy'])
  

  model.fit(
    train,
    validation_data=valid,
    epochs= num_epochs
  )

  model.save('models/clf/Model.keras')

  return model


train_data, valid_data = prep_datasets(path='dataset/animal-image-classification-dataset/Training Data/Training Data',valid_size = 0.25)
train_clf(train = train_data, valid = valid_data)




