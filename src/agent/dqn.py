import random
import numpy as np
import tensorflow as tf
from agent.model import DQNModel
from agent.replay_buffer import ReplayMemory
from config import STATE_SHAPE, NUM_ACTIONS, LEARNING_RATE, GAMMA, BATCH_SIZE, REPLAY_BUFFER_SIZE, EPSILON_START, EPSILON_MIN, EPSILON_DECAY, TARGET_UPDATE_FREQUENCY


class DQNAgent:
    def __init__(self, state_shape=STATE_SHAPE, num_actions=NUM_ACTIONS):
        self.state_shape = state_shape
        self.num_actions = num_actions

        self.gamma = GAMMA

        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        self.batch_size = BATCH_SIZE

        self.target_update_frequency = TARGET_UPDATE_FREQUENCY

        self.training_steps = 0

        self.memory = ReplayMemory(REPLAY_BUFFER_SIZE)

        self.q_network = DQNModel(
            input_shape = self.state_shape,
            num_actions = self.num_actions,
            learning_rate = LEARNING_RATE
        )

        self.target_network = DQNModel(
            input_shape = self.state_shape,
            num_actions = self.num_actions,
            learning_rate = LEARNING_RATE
        )

        self.update_target_network()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        state_tensor = tf.expand_dims(state_tensor, axis=0)
        
        q_values = self.q_network.model(state_tensor, verbose=0)

        return np.argmax(q_values.numpy()[0])
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update_target_network(self):
        self.target_network.model.set_weights(self.q_network.model.get_weights())

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        dones = tf.cast(dones, tf.float32)
        rewards = tf.cast(rewards, tf.float32)
        
        next_q_values = self.target_network.model(next_states, training= False)
        max_next_q = tf.reduce_max(next_q_values, axis =1)

        targets = rewards + (1.0 - dones) * self.gamma * max_next_q

        with tf.GradientTape() as tape:
            current_q_values = self.q_network.model(states, training=True)
            
            action_masks = tf.one_hot(actions, self.num_actions)
            predicted_q_values = tf.reduce_sum(current_q_values * action_masks, axis=1)
            
            ## Mean Squared Error Loss
            loss = tf.keras.losses.MSE(targets, predicted_q_values)

        gradients = tape.gradient(loss, self.q_network.model.trainable_variables)
        self.q_network.optimizer.apply_gradients(zip(gradients, self.q_network.model.trainable_variables))


    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        self._train_step(states, actions, rewards, next_states, dones)

        self.training_steps += 1

        if self.training_steps % self.target_update_frequency == 0 :
            self.update_target_network()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)    
                

    
    def save_model(self, path):
        self.q_network.model.save(path)

    def load_model(self, path):
        self.q_network.model = tf.keras.models.load_model(path)

        self.update_target_network()
