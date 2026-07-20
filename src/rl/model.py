import tensorflow as tf
from tensorflow.keras import layers, models
from config import STATE_SHAPE, NUM_ACTIONS, LEARNING_RATE

class DQNModel :
    def __init__(self, input_shape, num_actions, learning_rate):
        self.input_shape = STATE_SHAPE
        self.num_actions  = NUM_ACTIONS
        self.learning_rate = LEARNING_RATE

        self.model = self.build_model()

    def build_model(self):
            model = models.Sequential()
            model.add(layers.Conv2D(filters=32, kernel_size =  (8, 8), strides=(4, 4), activation='relu', input_shape=self.input_shape))
            model.add(layers.Conv2D(filters = 64, kernel_size = (4, 4), strides=(2, 2), activation='relu'))
            model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), strides=(1, 1), activation='relu'))
            model.add(layers.Flatten())
            model.add(layers.Dense(512, activation='relu'))
            model.add(layers.Dense(self.num_actions, activation='linear'))
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')

            return model   