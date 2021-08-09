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

        self.input_size = input_size
        self.output_size = output_size

        # Hyperparams
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Model
        x_in = tf.keras.Input([self.input_size,])
        x = tf.keras.layers.Dense(24, activation='relu')(x_in)
        x = tf.keras.layers.Dense(24, activation='relu')(x)
        x = tf.keras.layers.Dense(self.output_size, activation='linear')(x)
        self.model = tf.keras.Model(inputs=x_in, outputs=x, name='DQN')
        self.optimizer = tf.keras.optimizers.Adam(self.alpha)
        #self.model.summary()

        # Replay Memory [(s0, a0, r1, s1, done), (s1, a1, r2, s2, done)...]
        self.replay_memory = deque(maxlen=replay_size)
        self.batch_size = batch_size

    def policy(self, state):
        if random.random() > self.epsilon:
            return np.argmax(self.model.predict(state))
        else:
            return self.env.action_space.sample()
        
    def run(self, episodes, name=None):
        if name:
            with open('logs/dqn/'+name+'.csv', 'w+') as f:
                f.write('episode,reward,loss,len,e\n')
        for ep in range(episodes):
            episode_loss = 0
            episode_reward = 0
            episode_len = 0
            
            state = self.env.reset()
            state = np.reshape(state, [1, len(state)])

            # Run Episode
            done = False
            while not done:
                action = self.policy(state)

                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, len(next_state)])

                self.replay_memory.append((state, action, reward, next_state, done))
                loss = self.replay()

                episode_reward += reward
                episode_loss += loss
                episode_len += 1

                state = next_state
            print('episode:', ep, 'reward:', episode_reward, 'loss:', episode_loss.numpy(), 'len:', episode_len, 'e:', self.epsilon)

            # Logging
            if name:
                with open('logs/dqn/'+name+'.csv', 'a') as f:
                    f.write(str(ep)+',')
                    f.write(str(episode_reward)+',')
                    f.write(str(episode_loss.numpy())+',')
                    f.write(str(episode_len)+',')
                    f.write(str(self.epsilon)+'\n')

            # Adjust epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def replay(self):
            batch = random.sample(self.replay_memory, min(self.batch_size, len(self.replay_memory)))
            states, actions, rewards, next_states, dones = zip(*batch)

            with tf.GradientTape() as tape:
                Qs_actual = rewards + self.gamma*np.max(self.model(np.vstack(next_states)))*(not dones)
                Qs_selected = tf.reduce_sum(self.model(np.vstack(states))*tf.one_hot(actions, self.output_size), axis=1)
                loss = tf.math.reduce_mean(tf.square(Qs_actual-Qs_selected))

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            return loss
