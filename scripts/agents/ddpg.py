from collections import deque
import random
import os
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

tf.compat.v1.enable_eager_execution()

class DDPG():
    def __init__(self, env, state_dim, action_dim, action_bounds, replay_size=50000, batch_size=64,
            gamma=0.99, actor_alpha=0.001, critic_alpha=0.002, tau=0.005):

        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.replay_size = replay_size
        self.batch_size = batch_size

        # Hyperparams
        self.gamma = gamma
        self.actor_alpha = actor_alpha
        self.critic_alpha = critic_alpha
        self.tau = tau

        self.noise = OUActionNoise(mu=np.zeros(1), sigma=0.2*np.ones(1))

        # Actor and critic models
        self.actor_model = self.create_actor()
        self.actor_target_model = self.create_actor()
        self.actor_target_model.set_weights(self.actor_model.get_weights())
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_alpha)

        self.critic_model = self.create_critic()
        self.critic_target_model = self.create_critic()
        self.critic_target_model.set_weights(self.critic_model.get_weights())
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_alpha)

        #print(self.actor_model.summary())
        #print(self.critic_model.summary())

        # Replay Memory [(s0, a0, r1, s1, done), (s1, a1, r2, s2, done)...]
        self.replay_memory = deque(maxlen=replay_size)

    def policy(self, state):
        action = self.actor_model(np.vstack(state))
        action = action.numpy() + self.noise()
        action = np.clip(action, -self.action_bounds, self.action_bounds)
        action = np.squeeze(action)
        return action

    def run(self, episodes, name=None):
        if name:
            with open('logs/ddpg/'+name+'.csv', 'w+') as f:
                f.write('episode,reward,loss\n')

        episode_rewards = [0]*episodes

        for ep in range(episodes):
            prev_state = self.env.reset()
            prev_state = np.reshape(prev_state, [1, len(prev_state)])

            done = False
            episode_critic_loss = 0
            episode_actor_loss = 0
            while not done:
                action = self.policy(prev_state)
                state, reward, done, _ = self.env.step(action)
                state = np.reshape(state, [1, len(state)])

                self.replay_memory.append((prev_state, action, reward, state, done))
                critic_loss, actor_loss = self.replay()

                self.update_target(self.actor_target_model.variables, self.actor_model.variables)
                self.update_target(self.critic_target_model.variables, self.critic_model.variables)

                episode_rewards[ep] += reward
                episode_critic_loss += critic_loss
                episode_actor_loss += actor_loss

                prev_state = state
            
            # Logging
            if name:
                with open('logs/ddpg/'+name+'.csv', 'a') as f:
                    f.write(str(ep)+',')
                    f.write(str(episode_rewards[ep])+',')
                    f.write(str(critic_loss.numpy())+'\n')

            print('ep:', ep, 'ave reward:', np.mean(episode_rewards[ep-40:ep]), episode_critic_loss.numpy(), episode_actor_loss.numpy())

    def replay(self):
        batch = random.sample(self.replay_memory, min(self.batch_size, len(self.replay_memory)))
        states, actions, rewards, next_states, dones = zip(*batch)
        states = tf.convert_to_tensor(np.vstack(states), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.vstack(actions), dtype=tf.float32)
        rewards = tf.convert_to_tensor(np.vstack(rewards), dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.vstack(next_states), dtype=tf.float32)

        return self.compute_grads(states, actions, rewards, next_states)


    @tf.function
    def compute_grads(self, states, actions, rewards, next_states):
        with tf.GradientTape() as tape:
            actor_target_actions = self.actor_target_model(next_states)
            y = rewards + self.gamma*self.critic_target_model([next_states, actor_target_actions])
            critic_Q = self.critic_model([states, actions])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_Q))

        critic_grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actor_actions = self.actor_model(states)
            critic_Q = self.critic_model([states, actor_actions])
            actor_loss = -tf.math.reduce_mean(critic_Q)

        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

        return critic_loss, actor_loss

    @tf.function
    def update_target(self, target_vars, vars):
        for tv, v in zip(target_vars, vars):
            tv.assign(v*self.tau + tv*(1-self.tau))

    def create_actor(self):
        state_in = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(256, activation='relu')(state_in)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(self.action_dim, activation='tanh',
                kernel_initializer=tf.random_uniform_initializer(-0.003, 0.003))(x)

        # must scale action_out between -action_bounds and action_bounds
        action_out = tf.multiply(x, self.action_bounds)

        return tf.keras.Model(inputs=state_in, outputs=action_out)

    def create_critic(self):
        state_in = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(16, activation='relu')(state_in)
        x = layers.Dense(32, activation='relu')(x)

        action_in = layers.Input(shape=(self.action_dim))
        temp = layers.Dense(32, activation='relu')(action_in)

        x = layers.Concatenate()([x, temp])

        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        Q_out = layers.Dense(1)(x)

        return tf.keras.Model(inputs=[state_in, action_in], outputs=Q_out)

class OUActionNoise():
    def __init__(self, mu, sigma, theta=0.15, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta*(self.mu-self.x_prev)*self.dt + \
            self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)



#env = gym.make('Pendulum-v0')
#o_space = len(env.observation_space.sample())
#a_space = env.action_space.shape[0]
#a_high = env.action_space.high[0]
#
#agent = DDPG(env, o_space, a_space, a_high)
#agent.run(100)
