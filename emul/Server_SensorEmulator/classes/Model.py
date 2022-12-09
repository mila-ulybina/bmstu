import os
import numpy as np
import tensorflow as tf

#Описание модели нейронной сети
class Model(object):
    # Количество измерительных каналов
    NB_SENSOR_CHANNELS = 113

    # Количество классов распознаваемых движений
    NUM_CLASSES = 18

    # Длина скользящего окна (сегментация данных)
    SLIDING_WINDOW_LENGTH = 24

    # Длина входной последовательности, после свертки
    FINAL_SEQUENCE_LENGTH = 8

    # Шаг скользящего окна
    SLIDING_WINDOW_STEP = 12

    # Размер обучающих пакетов
    BATCH_SIZE = 100

    # Число фильтров в сверточных слоях
    NUM_FILTERS = 64

    # Размер фильтров в сверточных слоях
    FILTER_SIZE = 5

    # Число нейронов в LSTM слоях
    NUM_UNITS_LSTM = 128

    def __init__(self, static_path, model_path):
        self.model = self.create_model()
        self.model.load_weights(os.path.join(static_path, model_path))

    def create_model(self):
        inputs = tf.keras.layers.Input((self.NB_SENSOR_CHANNELS, self.SLIDING_WINDOW_LENGTH, 1))#, BATCH_SIZE) #[(None, 113, 24, 1)]
        conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,self.FILTER_SIZE), activation='relu', data_format='channels_last')(inputs) # channels_first corresponds to inputs with shape (batch_size, channels,height, width)
        conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,self.FILTER_SIZE), activation='relu', data_format='channels_last')(conv1)
        conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,self.FILTER_SIZE), activation='relu', data_format='channels_last')(conv2)
        conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,self.FILTER_SIZE), activation='relu', data_format='channels_last')(conv3)
        shuffle = tf.keras.layers.Permute((2, 3, 1))(conv4)
        reshape1 = tf.keras.layers.Reshape((int(shuffle.shape[1]), int(shuffle.shape[2]) * int(shuffle.shape[3])))(shuffle)
        lstm1 = tf.keras.layers.LSTM(self.NUM_UNITS_LSTM, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(reshape1)
        lstm2 = tf.keras.layers.LSTM(self.NUM_UNITS_LSTM, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(lstm1)
        reshape2 = tf.keras.layers.Lambda(self.backend_reshape, output_shape=(self.NUM_UNITS_LSTM,))(lstm2)
        softmax = tf.keras.layers.Dense(self.NUM_CLASSES, activation='softmax')(reshape2)
        reshape3 = tf.keras.layers.Lambda(self.backend_reshape_back, output_shape=(self.FINAL_SEQUENCE_LENGTH, self.NUM_CLASSES,))(softmax)
        output = tf.keras.layers.Lambda(lambda x: x[:,-1,:], output_shape=(self.NUM_CLASSES,))(reshape3)
        model = tf.keras.models.Model(inputs, output)

        return model

    def backend_reshape(self, x):
        return tf.keras.backend.reshape(x, (-1, self.NUM_UNITS_LSTM))

    def backend_reshape_back(self, x):
        return tf.keras.backend.reshape(x, (-1, self.FINAL_SEQUENCE_LENGTH, self.NUM_CLASSES))

    def predict(self, X_in):
        return np.argmax(self.model.predict(X_in), axis=1)