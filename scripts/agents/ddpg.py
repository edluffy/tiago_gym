from collections import deque
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class DDPG():
    def __init__(self, env, state_dim, action_dim, action_bound, replay_size, batch_size,
            gamma, alpha, tau):

        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.replay_size = replay_size
        self.batch_size = batch_size

        # Hyperparams
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau

        self.noise = OUActionNoise()

        # Actor and critic models
        self.actor_model = create_actor()
        self.actor_target_model = create_actor()
        self.critic_model = create_critic()
        self.critic_target_model = create_critic()

        self.optimizer = tf.keras.optimizers.Adam(self.alpha)

        # Replay Memory [(s0, a0, r1, s1, done), (s1, a1, r2, s2, done)...]
        self.replay_memory = deque(maxlen=replay_size)

    def policy(self, state):
        action = self.actor_model(state)
        action = actions + noise
        action = np.clip(action, -self.action_bound, self.action_bound)
        return action

    def run(self, episodes):
        for ep in range(episodes):

            state = self.env.reset()
            done = False
            while not done:
                action = self.policy(prev_state)
                state, reward, done, _ = self.env.step(action)

                self.replay_memory.append((prev_state, action, reward, state, done))
                self.replay()

                self.update_target(actor_target_model.variables, actor_model.variables)
                self.update_target(critic_target_model.variables, critic_model.variables)

                prev_state = state

    @tf.function
    def replay(self):
        batch = random.sample(self.replay_memory, min(self.batch_size, len(self.replay_memory)))
        states, actions, rewards, next_states, dones = zip(*batch)

        with tf.GradientTape() as tape:
            actor_target_actions = self.actor_target_model(next_states)
            y = rewards + gamma*self.critic_target_model([next_states, actor_target_actions])
            critic_Q = self.critic_model([states, actions])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_Q))

        critic_grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actor_actions = self.actor_model(states)
            critic_Q = self.critic_model([states, actions])
            actor_loss = -tf.math.reduce_mean(critic_Q)

        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

    @tf.function
    def update_target(target_vars, vars):
        for tv, v in zip(target_vars, vars):
            tv.assign(v*self.tau + tv*(1-self.tau))

    def create_actor(self):
        state_in = keras.input(shape=[None,self.state_dim])

        x = layers.Dense(400)(state_in)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dense(300)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Dense(self.action_dim,
                kernel_initializer=tf.random_uniform_initializer(-0.003, 0.003))(x)
        x = layers.Activation('tanh')(x)

        # must scale action_out between -action_bound and action_bound
        action_out = tf.multiply(x, self.action_bound)

        return tf.keras.Model(inputs=state_in, outputs=action_out)

    def create_critic(self):
        state_in = keras.input(shape=[None,self.state_dim])
        x = layers.Dense(400)(state_in)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        action_in = keras.input(shape=[None,self.action_dim])
        temp = layers.Dense(400)(action_in)

        x = layers.Concatenate()([x, temp])
        x = layers.Dense(300)(x)
        x = layers.Activation('relu')(x)

        Q_out = layers.Dense(1, kernel_initializer=tf.random_uniform_initializer(-0.003, 0.003))(x)

        return tf.keras.Model(inputs=[state_in, action_in], outputs=Q_out)

class OUActionNoise():
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
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
