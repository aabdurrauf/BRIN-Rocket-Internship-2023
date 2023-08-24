#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from numpy.core.numeric import False_
import pandas as pd
import tensorflow as tf

from control_and_ai.DDPG.ddpg import DDPG
from control_and_ai.DDPG.utils import Utils
from control_and_ai.DDPG.exploration import OUPolicy

from constants import *
from constants import DEGTORAD
from environments.rocketlander_ga_psi import RocketLander, get_state_sample

action_bounds = [1, 1, 15 * DEGTORAD]

eps = []
eps.append(OUPolicy(0, 0.2, 0.4))
eps.append(OUPolicy(0, 0.2, 0.4))
eps.append(OUPolicy(0, 0.2, 0.4))

genes = [1, 0, 0, 0, 0, 0, 0, 0, 1, 1]

simulation_settings = {'Side Engines': True,
                       'Clouds': True,
                       'Vectorized Nozzle': False,
                       'Graph': False,
                       'Render': False,
                       'Starting Y-Pos Constant': 1,
                       'Initial Force': 'random',
                       'Rows': 1,
                       'Columns': 2,
                       'Episodes': 500,
                       'Genes': genes}
env = RocketLander(simulation_settings)

# Set both line below to False if you want to contniue training from a saved checkpoint
RETRAIN = False  # Restore weights if False
# TEST = False #Test the model
TEST = True  # Test the model

NUM_EPISODES = 100
SAVE_REWARD = True  # Export reward log as .xlsx
NAME = "ind17_psiON"  # Model name

model_dir = 'D:/Programming/Python/rocketlander/control_and_ai/DDPG/trained_models/' + NAME
os.makedirs("model_dir", exist_ok=True)

with tf.device('/cpu:0'):
    agent = DDPG(
        action_bounds,
        eps,
        env.observation_space.shape[0],  # for first model
        actor_learning_rate=0.001,
        critic_learning_rate=0.01,
        retrain=RETRAIN,
        log_dir="./logs",
        model_dir=model_dir,
        batch_size=100,
        gamma=0.99)


# In[2]:


def train(env, agent):
    obs_size = env.observation_space.shape[0]

    util = Utils()
    state_samples = get_state_sample(samples=5000, genes=genes, normal_state=True)

    util.create_normalizer(state_sample=state_samples)
    if SAVE_REWARD:
        rew = []
        ep = []

    for episode in range(1, NUM_EPISODES + 1):
        old_state = None
        done = False
        total_reward = 0

        state = env.reset()
        state = util.normalize(state)
        max_steps = 500

        left_or_right_barge_movement = np.random.randint(0, 2)

        for t in range(max_steps):  # env.spec.max_episode_steps
            old_state = state
            # infer an action
            action = agent.get_action(np.reshape(state, (1, obs_size)), not TEST)

            # take it
            state, reward, done, _ = env.step(action[0])
            state = util.normalize(state)
            total_reward += reward

            if state[LEFT_GROUND_CONTACT] == 0 and state[RIGHT_GROUND_CONTACT] == 0:
                # env.move_barge_randomly(epsilon, left_or_right_barge_movement)
                env.apply_random_x_disturbance(epsilon=0.005, left_or_right=left_or_right_barge_movement)
                env.apply_random_y_disturbance(epsilon=0.005)

            if TEST:
                # update q vals
                agent.update(old_state, action[0], np.array(reward), state, done)

            if done:
                break

        agent.log_data(total_reward, episode)

        if episode % 50 == 0 and not TEST:
            print('Saved model at episode', episode)
            agent.save_model(episode)
        if SAVE_REWARD:
            rew.append(total_reward)
            ep.append(episode)
        print("Episode:\t{0}\tReward:\t{1}".format(episode, total_reward))

    if SAVE_REWARD:
        os.makedirs(model_dir, exist_ok=True)
        reward_data = pd.DataFrame(list(zip(ep, rew)), columns=['episode', 'reward'])
        with pd.ExcelWriter(model_dir + f"/DDPG_eps-rewards_{NAME}_{rew[-1]}_{len(ep)}.xlsx") as writer:
            reward_data.to_excel(writer, sheet_name=f"{NAME}_eps-rewards")


# In[3]:


with tf.device('/cpu:0'):
    train(env, agent)
