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
        model.add(layers.Conv2D(32, (8, 8), padding='same',strides=(4, 4))) 
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Conv2D(64, (4, 4),strides=(2, 2),  padding='same'))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(64, (3, 3),strides=(1, 1),  padding='same'))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Activation('relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
      
        if hasattr(self.num_actions, 'n'):
            clean_units = int(self.num_actions.n)
        else:
            clean_units = int(self.num_actions)
        
        model.add(layers.Dense(clean_units, activation='linear'))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        return model 


class DuelingDQNModel:
    def __init__(self, input_shape, num_actions, learning_rate):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate

        self.model = self.build_model()

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)

        x = layers.Conv2D(filters = 32, kernel_size=(8,8), strides=(4, 4), padding='same', activation='relu')(inputs)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(filters = 64, kernel_size = (4, 4), strides = (2, 2), padding='same', activation = 'relu')(x)
        x = layers.MaxPooling2D(pool_size =(2, 2))(x)
        x = layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding='same', activation = 'relu')(x)
        x = layers.MaxPooling2D(pool_size = (2, 2))(x)
        x = layers.Flatten()(x)
        shared = layers.Dense(512, activation='relu')(x)
        
        value = layers.Dense(256, activation = 'relu')(shared)
        value = layers.Dense(1, name='value')(value)

        advantage = layers.Dense(256, activation='relu')(shared)
        advantage = layers.Dense(self.num_actions, name='advantage')(advantage)

        q_values = layers.Lambda(lambda x: x[0] + (x[1] - tf.reduce_mean(x[1], axis=1, keepdims=True)), names='Qvalues')([value, advantage])

        model = models.Model(inputs = inputs, outputs = q_values, name='DuelingDQN')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        return model


