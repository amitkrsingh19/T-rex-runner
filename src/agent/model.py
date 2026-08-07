import tensorflow as tf
from tensorflow.keras import layers, models


class DQNModel :
    def __init__(self, input_shape, num_actions, learning_rate):
        self.input_shape = input_shape
        self.num_actions  = num_actions
        self.learning_rate = learning_rate

        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Input(shape=self.input_shape))

        model.add(layers.Conv2D(32, (8, 8), padding='same',strides=(2, 2), activation='relu')) 
        model.add(layers.MaxPooling2D(pool_size=(2,2)))

        model.add(layers.Conv2D(64, (4, 4),strides=(2, 2),  padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))

        model.add(layers.Conv2D(64, (3, 3),strides=(1, 1),  padding='same', activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
      
        if hasattr(self.num_actions, 'n'):
            clean_units = int(self.num_actions.n)
        else:
            clean_units = int(self.num_actions)
        
        model.add(layers.Dense(clean_units, activation='linear'))

        model.summary()
        
        return model   