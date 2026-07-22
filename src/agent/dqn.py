import random
import numpy as np
import tensorflow as tf
from agent.model import DQNModel
from agent.replay_buffer import ReplayMemory
from config import STATE_SHAPE, NUM_ACTIONS, LEARNING_RATE, GAMMA, BATCH_SIZE, REPLAY_BUFFER_SIZE, EPSILON_START, EPSILON_MIN, EPSILON_DECAY, TARGET_UPDATE_FREQUENCY

class BaseAgent:
    def __init__(self, policy_network, target_network):
        self.epsilon = EPSILON_START
        self.memory = ReplayMemory(REPLAY_BUFFER_SIZE)
        self.num_actions = NUM_ACTIONS

        self.policy_network = policy_network
        self.target_network = target_network

        self.update_target_network()

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32) / 255.0  ## shape (83, 100, 4)
        state_tensor = tf.expand_dims(state_tensor, axis=0)     ## shape (1, 83, 100, 4)
        
        q_values = self.policy_network.model(state_tensor, training=False)

        return int(np.argmax(q_values.numpy()[0]))

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def save_model(self, path):
        self.policy_network.model.save(path)

    def load_model(self, path):
        self.policy_network.model = tf.keras.models.load_model(path)

        self.update_target_network()

    def update_target_network(self):
        self.target_network.model.set_weights(self.policy_network.model.get_weights())

class DQNAgent(BaseAgent):
    def __init__(self, num_actions, gamma=0.99, lr=1e-4):
        self.state_shape = STATE_SHAPE

        self.num_actions = num_actions
        self.gamma = GAMMA

        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

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

        super().__init__(self.q_network, self.target_network)      

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        dones = tf.cast(dones, tf.float32)
        rewards = tf.cast(rewards, tf.float32)

        states = states / 255.0
        next_states = next_states / 255.0
        
        next_q_values = self.target_network.model(next_states, training= False)
        max_next_q = tf.reduce_max(next_q_values, axis =1)

        targets = rewards + (1.0 - dones) * self.gamma * max_next_q

        with tf.GradientTape() as tape:
            current_q_values = self.q_network.model(states, training=True)
            
            action_masks = tf.one_hot(actions, self.num_actions)
            predicted_q_values = tf.reduce_sum(current_q_values * action_masks, axis=1)

            ## DEBUG
            #tf.print("target range:", tf.reduce_min(targets), tf.reduce_max(targets))
            #tf.print("predicted_q range:", tf.reduce_min(predicted_q_values), tf.reduce_max(predicted_q_values))

            ## Mean Squared Error Loss
            loss = tf.keras.losses.MSE(targets, predicted_q_values)
            mean_loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, self.q_network.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.model.trainable_variables))

        return mean_loss

    def train(self, batch_size = 4):
        if len(self.memory) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        loss_tensor = self._train_step(states, actions, rewards, next_states, dones)

        loss = loss_tensor.numpy()

        return float(loss)        