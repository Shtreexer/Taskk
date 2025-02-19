from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint


class MnistClassifierInterface():


  def __init__(self):

    self.model_name = None

    self.model_list ={'rf': MnistClassifierInterface.rf(),
                 'nn': MnistClassifierInterface.nn(),
                  'cnn' : MnistClassifierInterface.cnn()}

  def __call__(self, model_name):

    self.model_name = model_name

    self.model = self.model_list[self.model_name]


  def train(self, paths = None):

    self.model(paths=paths)
    self.model.train()

  def predict(self, data = None):
    res = self.model.predict(data)

    return res








  
  class rf():

    def __init__(self):

      self.paths = None

      self.train_data = None
      self.valid_data = None
      self.model = RandomForestClassifier(n_estimators=100)

      self.history = None
      self.res = None

    def __call__(self, paths = None):

      self.paths = paths

      self.train_data = pd.read_csv(paths['train'])
      self.train_data.columns = [str(x) for x in np.arange(0, len(self.train_data.columns))]
      self.valid_data = pd.read_csv(paths['valid'])
      self.valid_data.columns = [str(x) for x in np.arange(0, len(self.valid_data.columns))]

    def train(self):

      train_X, train_y = self.train_data.iloc[:,1:], self.train_data.iloc[:,0] 
      valid_X, valid_y = self.valid_data.iloc[:,1:], self.valid_data.iloc[:,0] 

      self.model.fit(train_X, train_y)
      y_pred = self.model.predict(valid_X)

      accuracy = accuracy_score(valid_y, y_pred)
      self.history = accuracy


      return self.history



    def predict(self, data = None):
      data.columns = [str(x) for x in np.arange(0, len(data.columns))]
      data = data.iloc[:100,1:]

      self.res = self.model.predict(data)

      return {'data': tf.data.Dataset.from_tensor_slices(data.to_numpy()),
              'pred': self.res}


  class nn():

    def __init__(self):

      self.paths = None

      self.train_data = None
      self.valid_data = None
      self.model = None

      self.history = None
      self.res = None

    def __call__(self, paths = None):

      self.train_data = pd.read_csv(paths['train'])
      self.train_data.columns = [str(x) for x in np.arange(0, len(self.train_data.columns))]
      self.valid_data = pd.read_csv(paths['valid'])
      self.valid_data.columns = [str(x) for x in np.arange(0, len(self.valid_data.columns))]

      def convert_to_tf(data):

        data = data.to_numpy()
        lb = LabelBinarizer()
        image = data[:,1:] / 255
        label = lb.fit_transform(data[:,0])
    
        return tf.data.Dataset.from_tensor_slices((image, label))

      self.train_data = convert_to_tf(self.train_data)
      self.valid_data = convert_to_tf(self.valid_data)

      self.train_data = self.train_data.batch(32).prefetch(tf.data.AUTOTUNE)
      self.valid_data = self.valid_data.batch(32).prefetch(tf.data.AUTOTUNE)


    def train(self):

      self.model = tf.keras.Sequential([
      
      tf.keras.layers.Input(shape = (784,)),
      tf.keras.layers.Dense(784, activation="sigmoid"),
      tf.keras.layers.Dense(256, activation="sigmoid"),
      tf.keras.layers.Dense(128, activation="sigmoid"),
      tf.keras.layers.Dense(10, activation="softmax")
      
      ])


      self.model.compile(
          
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics = ['accuracy']

            )
      
      self.history = self.model.fit(self.train_data,
                              validation_data = self.valid_data,
                              epochs = 3,
                              callbacks = [ModelCheckpoint(filepath='models/FF_model/checkpoint.model.keras',
                              monitor='val_accuracy',
                              mode='max',
                              save_best_only=True)])
      

    def predict(self, data = None):

      '''data format: DataFrame, columns: 0-783: numeric '''

      data.columns = [str(x) for x in np.arange(0, len(data.columns))]
      data = tf.data.Dataset.from_tensor_slices((data.to_numpy()))


      result = self.model.predict(data.batch(32))
      self.result = {'data': data,
          'pred':np.array([x.argmax(axis = 0) for x in (np.exp(x)/sum(np.exp(x)) for x in result)])}

      return self.result

  class cnn():

    def __init__(self):

      self.paths = None

      self.train_data = None
      self.valid_data = None
      self.model = None

      self.history = None
      self.res = None



    def __call__(self, paths = None):

      self.train_data = pd.read_csv(paths['train'])
      self.train_data.columns = [str(x) for x in np.arange(0, len(self.train_data.columns))]
      self.valid_data = pd.read_csv(paths['valid'])
      self.valid_data.columns = [str(x) for x in np.arange(0, len(self.valid_data.columns))]

      def convert_to_tf(data):

        data = data.to_numpy()
        lb = LabelBinarizer()
        image = data[:,1:]
        label = lb.fit_transform(data[:,0])
    
        return tf.data.Dataset.from_tensor_slices((image, label))

      def prep_for_conv(data, lbl):

        data = data / 255
        return (tf.reshape(data, shape = (28, 28, 1)), lbl)


      self.train_data = convert_to_tf(self.train_data)
      self.valid_data = convert_to_tf(self.valid_data)


      self.train_data = self.train_data.map(prep_for_conv)
      self.valid_data = self.valid_data.map(prep_for_conv)

      self.train_data = self.train_data.batch(32).prefetch(tf.data.AUTOTUNE)
      self.valid_data = self.valid_data.batch(32).prefetch(tf.data.AUTOTUNE)


    def train(self):

      self.model = tf.keras.Sequential([
                        tf.keras.layers.Input(shape=(28, 28, 1)),
                        tf.keras.layers.Conv2D(28, (3, 3),  activation='relu', input_shape=(28, 28, 1)),
                        tf.keras.layers.MaxPool2D((2, 2)),
                        tf.keras.layers.Conv2D(56, (3, 3),  activation='relu'),
                        tf.keras.layers.MaxPool2D((2, 2)),
                        tf.keras.layers.Conv2D(56, (3, 3),  activation='relu'),
                        tf.keras.layers.Flatten(),

                        tf.keras.layers.Dense(56, activation = 'relu'),
                        tf.keras.layers.Dense(10)

                        ])
      
      self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics = ['accuracy'])
      

      self.history = self.model.fit(self.train_data,
        validation_data = self.valid_data,
        epochs = 3,
        callbacks = [ModelCheckpoint(filepath='models/CNN_model/checkpoint.model.keras',
                                    monitor='val_accuracy',
                                    mode='max',
                                    save_best_only=True)])
      

    def predict(self, data = None):

      data = tf.data.Dataset.from_tensor_slices([x.reshape((28, 28, 1)) for x in (np.array(data)/255)]).batch(32)
      result = self.model.predict(data)
      self.result = {'data': data,
                    'pred': np.array([x.argmax(axis = 0) for x in (np.exp(x)/sum(np.exp(x)) for x in result)])}

      return self.result




  


    