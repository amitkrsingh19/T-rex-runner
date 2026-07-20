import random
import numpy as np
import tensorflow as tf
from model import DQNModel
from memory import ReplayMemory
from config import STATE_SHAPE, NUM_ACTIONS, LEARNING_RATE, GAMMA, BATCH_SIZE, REPLAY_BUFFER_SIZE, EPSILON_START, EPSILON_MIN, EPSILON_DECAY

class DQNAgent:
    def __init__(self, state_shape=STATE_SHAPE, num_actions=NUM_ACTIONS):
        self.state_shape = state_shape
        self.num_actions = num_actions

        self.gamma = GAMMA

        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        self.batch_size = BATCH_SIZE

        self.memory = ReplayMemory(REPLAY_BUFFER_SIZE)

        self.model = DQNModel(
            input_shape = self.state_shape,
            num_actions = self.num_actions,
            learning_rate = LEARNING_RATE
        )

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        
        state = np.expand_dims(state, axis=0)
        
        q_values = self.model.model.predict(state, verbose=0)

        return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

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

        current_q_values = self.model.model.predict(states, verbose=0)
        next_q_values = self.model.model.predict(next_states, verbose=0)

        target_q_values = current_q_values.copy()

        for i in range(self.batch_size):
            if dones[i]:
                target = rewards[i]
            else:
                target = rewards[i] + self.gamma * np.max(next_q_values[i])
            
            target_q_values[i][actions[i]] = target

        self.model.model.fit(states, target_q_values, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    
    def save_model(self, path):
        self.model.model.save(path)

    def load_model(self, path):
        self.model.model = tf.keras.models.load_model(path)
