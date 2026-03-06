
import numpy as np
import random
from collections import deque
import gymnasium as gym
import tensorflow as tf
from tensorflow.keras import Model, layers
import os


env = gym.make('CartPole-v1')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

print("State Size: ", state_size)
print("Action Size: ", action_size)

class DQN(Model):
    def __init__(self, action_size, **kwargs):
        super(DQN, self).__init__(**kwargs)
        self.action_size = action_size
        self.d1 = layers.Dense(24, activation='relu', name='d1')
        self.d2 = layers.Dense(24, activation='relu', name='d2')
        self.d3 = layers.Dense(action_size, activation='linear', name='d3')

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)

    # Configs for loading the saved model file later on
    def get_config(self):
        config = super(DQN, self).get_config()
        config.update({"action_size": self.action_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



#############important###############

# Replay memory deque
memory = deque(maxlen=2000)

class Agent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _build_model(self):
        return DQN(self.action_size)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model(np.array([state]))
        return np.argmax(q_values[0].numpy())

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        # Load the saved model from the specified filepath
        self.model = tf.keras.models.load_model(filepath, custom_objects={"DQN": DQN})
        self.target_model = tf.keras.models.load_model(filepath, custom_objects={"DQN": DQN})
        
    def replay(self, batch_size):
        minibatch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            with tf.GradientTape() as tape:
                q_values = self.model(np.array([state]), training=True)
                q_value = q_values[0][action]

                if done:
                    target = reward
                else:
                    next_action = np.argmax(self.model(np.array([next_state]))[0].numpy())
                    t = self.target_model(np.array([next_state]))[0][next_action]
                    target = reward + self.gamma * t

                loss = tf.reduce_mean(tf.square(target - q_value))

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
