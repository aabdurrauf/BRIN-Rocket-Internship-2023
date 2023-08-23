#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
# import time

import numpy as np
# from numpy.core.numeric import False_
import pandas as pd
import tensorflow as tf

import xpc
from control_and_ai.DDPG.ddpg import DDPG
from control_and_ai.DDPG.utils import Utils
from control_and_ai.DDPG.exploration import OUPolicy

# from constants import *
# from constants import DEGTORAD
# from environments.rocketlander_ga_psi import RocketLander, get_state_sample
from environments.xplane import XPlane11, get_state_sample

action_bounds = [1, 1, 1, 1]

eps = []
eps.append(OUPolicy(0, 0.2, 0.4))
eps.append(OUPolicy(0, 0.2, 0.4))
eps.append(OUPolicy(0, 0.2, 0.4))
eps.append(OUPolicy(0.5, 0.2, 0.4))

# genes = [0, 0, 0, 0, 1, 0, 0, 0, 1, 1]

# simulation_settings = {'Side Engines': True,
#                        'Clouds': True,
#                        'Vectorized Nozzle': False,
#                        'Graph': False,
#                        'Render': False,
#                        'Starting Y-Pos Constant': 1,
#                        'Initial Force': 'random',
#                        'Rows': 1,
#                        'Columns': 2,
#                        'Episodes': 500,
#                        'Genes': genes}
# env = RocketLander(simulation_settings)
client = xpc.XPlaneConnect()
env = XPlane11(client)

# Set both line below to False if you want to continue training from a saved checkpoint
RETRAIN = True  # Restore weights if False
TEST = False  # Test the model
# TEST = Test # Test the model

NUM_EPISODES = 30
SAVE_MOD = 1
SAVE_REWARD = True  # Export reward log as .xlsx
TRAIN_NUM = "32"
NAME = "xplane-train-" + TRAIN_NUM  # Model name

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
    state_samples = get_state_sample(env=env, xpc_client=client, samples=500, normal_state=True)
    print("finished gathering state samples")
    # time.sleep(15)

    util.create_normalizer(state_sample=state_samples)
    if SAVE_REWARD:
        rew = []
        ep = []

    # the starship might still ih the air, wait until it crashes and reset itself
    # crash = 0
    # while not crash:
    #     try:
    #         crash = env.client.getDREF("sim/flightmodel2/misc/has_crashed")[0]
    #     except Exception as e:
    #         crash = 0
    #
    crash = 1
    while crash:
        try:
            crash = env.client.getDREF("sim/flightmodel2/misc/has_crashed")[0]
        except Exception as e:
            crash = 1
            env.client.clearBuffer()
        # time.sleep(0.01)

    # env.client.sendDREF("sim/flightmodel2/misc/has_crashed", 0)
    # time.sleep(2)

    for episode in range(1, NUM_EPISODES + 1):
        env.client.clearBuffer()
        old_state = None
        done = False
        total_reward = 0

        state = env.reset()
        state = util.normalize(state)
        max_steps = 500  # default max_steps = 500
        max_step_reached = False
        # time.sleep(2)

        for t in range(max_steps):
            old_state = state
            # infer an action
            action = agent.get_action(np.reshape(state, (1, obs_size)), not TEST)
            # print("action:", action)
            # take it
            env.client.clearBuffer()
            state, reward, done, _ = env.step(action[0])
            state = util.normalize(state)

            # penalty if max step is reached but have not landed
            if t >= max_steps:
                reward = -5000
                # max_step_reached = True

            # check again if crash happened but have not detected
            env.client.clearBuffer()
            try:
                crash = env.client.getDREF("sim/flightmodel2/misc/has_crashed")[0]
            except Exception as e:
                crash = 0
                env.client.clearBuffer()

            if crash == 1 and reward != -100000:
                print("starship crashed, detected outside step function")
                # print("crash 03:", self.crash)
                # done = True
                reward -= 100000
                done = True

            total_reward += reward
            # print("total reward:", total_reward)

            if not TEST:
                # update q vals
                agent.update(old_state, action[0], np.array(reward), state, done)

            if done:
                # print("Starship crashed/landed detected inside the train function")
                crash = 1
                while crash:
                    try:
                        crash = env.client.getDREF("sim/flightmodel2/misc/has_crashed")[0]
                    except Exception as e:
                        crash = 1
                        env.client.clearBuffer()

                # if crash > 0.0:
                #     time.sleep(20)

                break

        # if not max_step_reached:
        #     crash = 0
        #     while crash == 0:
        #         try:
        #             crash = env.client.getDREF("sim/flightmodel2/misc/has_crashed")
        #         except Exception as e:
        #             crash = 0
        #         time.sleep(1)

        agent.log_data(total_reward, episode)

        if episode % SAVE_MOD == 0 and not TEST:
            print('Saved model at episode', episode)
            agent.save_model(episode)
        if SAVE_REWARD:
            rew.append(total_reward)
            ep.append(episode)
        print("Episode:\t{0}\tReward:\t{1}".format(episode, total_reward))

        elev = []
        aile = []
        rudd = []
        thro = []

        alti = []
        posx = []
        posz = []
        vervel = []
        xvel = []
        zvel = []
        pitc = []
        roll = []
        yaw = []
        # pitr = []
        # rolr = []
        # yawr = []
        action_history = env.get_action_history()
        black_box = env.get_black_box()
        # black_box = env.
        for actions in action_history:
            elev.append(actions[0])
            aile.append(actions[1])
            rudd.append(actions[2])
            thro.append(actions[3])
        for value in black_box:
            alti.append(value[0])
            posx.append(value[1])
            posz.append(value[2])
            vervel.append(value[3])
            xvel.append(value[4])
            zvel.append(value[5])
            pitc.append(value[6])
            roll.append(value[7])
            yaw.append(value[8])
            # pitr.append(value[9])
            # rolr.append(value[10])
            # yawr.append(value[11])

        action_history = pd.DataFrame(list(zip(elev, aile, rudd, thro, alti, posx, posz, vervel,
                                               xvel, zvel, pitc, roll, yaw)),
                                      columns=['elevator', 'aileron', 'rudder', 'throttle', 'altitude', 'pos_x',
                                               'pos_z', 'ver. velocity', 'x velocity', 'z velocity', 'pitch',
                                               'roll', 'yaw'])
        if episode == 1:
            with pd.ExcelWriter(model_dir + f"/DDPG_action-history_{NAME}_test-{TRAIN_NUM}_{NUM_EPISODES}.xlsx") as writer:
                action_history.to_excel(writer, sheet_name=f"episode_{episode}")
        else:
            with pd.ExcelWriter(model_dir + f"/DDPG_action-history_{NAME}_test-{TRAIN_NUM}_{NUM_EPISODES}.xlsx", mode='a') as writer:
                action_history.to_excel(writer, sheet_name=f"episode_{episode}")
    # env.black_box
    if SAVE_REWARD:
        os.makedirs(model_dir, exist_ok=True)
        reward_data = pd.DataFrame(list(zip(ep, rew)), columns=['episode', 'reward'])
        with pd.ExcelWriter(model_dir + f"/DDPG_eps-rewards_{NAME}_{rew[-1]}_{len(ep)}.xlsx") as writer:
            reward_data.to_excel(writer, sheet_name=f"{NAME}_eps-rewards")


# In[3]:


with tf.device('/cpu:0'):
    train(env, agent)

env.close()
client.close()
