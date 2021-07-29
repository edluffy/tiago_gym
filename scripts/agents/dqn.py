from collections import deque
import tensorflow as tf
import numpy as np
import random
import gym

# Deep Q-Learning with Experience Replay
class DQN:
    def __init__(self, env, input_size, output_size, replay_size=2000, batch_size=32,
            gamma=0.9, alpha=0.0005, epsilon=1.0, epsilon_decay=0.97, epsilon_min=0.01):
        self.env = env

        # Hyperparams
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Model
        x_in = tf.keras.Input([input_size,])
        x = tf.keras.layers.Dense(24, activation='relu')(x_in)
        x = tf.keras.layers.Dense(24, activation='relu')(x)
        x = tf.keras.layers.Dense(output_size, activation='linear')(x)

        self.model = tf.keras.Model(inputs=x_in, outputs=x, name='Deep Q-Learning with Experience Replay')
        self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(self.alpha))
        self.model.summary()

        # Replay Memory [(s0, a0, r1, s1, done), (s1, a1, r2, s2, done)...]
        self.replay_memory = deque(maxlen=replay_size)
        self.batch_size = batch_size

    def policy(self, state):
        if random.random() > self.epsilon:
            return np.argmax(self.model.predict(state))
        else:
            return self.env.action_space.sample()
        
    def run(self, episodes):
        for ep in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, len(state)])

            # Run Episode
            done = False
            reward_sum = 0
            while not done:
                action = self.policy(state)

                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, len(next_state)])

                self.replay_memory.append((state, action, reward, next_state, done))

                state = next_state
                reward_sum += reward
            print('episode:', ep, 'rewards:', reward_sum, 'e:', self.epsilon)

            # Experience Replay
            batch = random.sample(self.replay_memory, min(self.batch_size, len(self.replay_memory)))

            for state, action, reward, next_state, done in batch:
                Qs = self.model.predict(state)[0]
                Qs_next= self.model.predict(next_state)[0]
                Qs[action] = reward if done else reward + self.gamma*np.amax(Qs_next)

                self.model.fit(state, np.array([Qs]), verbose=0)

            # Adjust epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
