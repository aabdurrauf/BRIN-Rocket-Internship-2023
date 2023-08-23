#!/usr/bin/env python
# coding: utf-8

# <h1>Test Program - PID - Constant Wind</h1>

# This test uses a PID controller for the rocket. Configured to be ran locally and <strong>gives constant wind disturbance during the flight</strong>.

# # Settings

# <h5>Simulation Config</h5>

# In[ ]:


NUM_EPISODES = 1 #Number of episodes to run

#Keep Nozzle Angle Constant to 0 degree (True) or not (False)
TURN_OFF_NOZZLE = False

#Wind Properties
SIMULATE_WIND = False #Simulate wind or not (True or False)
total_windstep = 50 #gives wind for total step
wind_dir = 'right' #wind direction : 'left' or 'right'
x_force = 5 # x-axis wind force in Newton
y_force = 0 # y-axis wind force in Newton

SAVE_EACH_EPS_LOG = True #Export each episode states & actions logs as .xlsx
SAVE_SUMMARY = True #Export summary logs as .xlsx (SAVE_EACH_EPS_LOG must be True)
SAVE_PLOT = True # Export plot for each episode (SAVE_EACH_EPS_LOG must be True)

##############################################Ignore this part#####################################################
genes = [1,1,1,1,1,1,1,1,1,1]
if len(genes) != 10:
  raise Exception("incorrect genes size")

if SIMULATE_WIND == False:
    wind_dir = 'none'
    
if TURN_OFF_NOZZLE:
    nozzle_toggle = 'OFF'
else:
    nozzle_toggle = 'ON'

test_name = f'PID_wind{wind_dir.capitalize()}_nozzle{nozzle_toggle}'
##############################################Ignore this part#####################################################


# # Functions

# In[ ]:


def plot(state_or_action,data, save_dir, episode, total_reward):
    #TEST RESULT: STATE
    data_df = data
    data_df['step'] = data_df.index
    titles = {'step':'Time','x_pos':'Miss Distance','y_pos':'Altitude','x_vel':'Horizontal Velocity','y_vel':'Vertical Velocity',
                'angle_deg':'Lateral Angle','angvel_deg':'Angular Velocity','remaining_fuel':'Remaining Fuel',
                'Fe':'Main Engine','Fs':'Side Thrusters','Psi_deg':'Nozzle Angle'}
    labels = {'step':'Time (s)','x_pos':'Miss Distance (m)','y_pos':'Altitude (m)','x_vel':'Horizontal Velocity (m/s)','y_vel':'Vertical Velocity (m/s)',
                'angle_deg':'Lateral Angle (deg)','angvel_deg':'Angular Velocity (deg/s)','remaining_fuel':'Remaining Fuel (kg)',
                'Fe':'Main Engine (N)','Fs':'Side Thrusters (N)','Psi_deg':'Nozzle Angle (deg)'}
    
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    matplotlib.use('Agg')
    plt.rcdefaults()
    
    fig, ax = plt.subplots()
    
    def main_plot(xx,yy):
      images_dir = save_dir
      if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        
      xdata = data_df[xx].to_numpy()
      ydata = data_df[yy].to_numpy()
      
      xlabel = labels.get(xx)
      ylabel = labels.get(yy)
      xtitle = titles.get(xx)
      ytitle = titles.get(yy)
      
      filename = f"{xx}_{yy}"
      title = f"PID\n{ytitle} against {xtitle}"
    
      ax.plot(xdata,ydata)
      if xx == 'x_pos' and yy == 'y_pos':
        ax.add_patch(Rectangle((-4.95, -1), 9.9, 1, facecolor = 'black', fill=True, alpha=0.9))
        ax.set_ylim([-1, 21])
      ax.set_xlabel(xlabel)
      ax.set_ylabel(ylabel)
      ax.set_title(title)
      ax.grid(linestyle='dotted', lw=1.5)
      # plt.savefig(f"{images_dir}" + filename + ".svg")
      plt.savefig(f"{images_dir}" + filename + ".png")
      plt.cla()

    if state_or_action == 'state': 
        main_plot('x_pos','y_pos')
        main_plot('step','x_pos')
        main_plot('step','y_pos')
        main_plot('step','x_vel')
        main_plot('step','y_vel')
        main_plot('step','angle_deg')
        main_plot('step','angvel_deg')
        main_plot('step','remaining_fuel')
    elif state_or_action == 'action':
        main_plot('step','Fe')
        main_plot('step','Fs')
        main_plot('step','Psi_deg')
        
    plt.close(fig)


# In[ ]:


def export_excel_append(df, file, sheet_name):
    if os.path.isfile(file):  # if file already exists append to existing file
        workbook = openpyxl.load_workbook(file)  # load workbook if already exists
        sheet = workbook[sheet_name]  # declare the active sheet 

        # append the dataframe results to the current excel file
        for row in dataframe_to_rows(df, header = False, index = False):
            sheet.append(row)
        workbook.save(file)  # save workbook
        workbook.close()  # close workbook
    else:  # create the excel file if doesn't already exist
        with pd.ExcelWriter(path = file, engine = 'openpyxl') as writer:
            df.to_excel(writer, index = False, sheet_name = sheet_name)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

def betterScale(x, xmin, xmax, a, b): # normalize to [a,b]
    return ( (x - xmin) / (xmax - xmin) * (b - a) ) + a

class Scaler(): # normalize to [0,1] with fit and transform
    def create_scaler(self, action_samples):
        self.scaler = self.fit_scaler(action_samples)
        
    def fit_scaler(self, action_samples):
        scaler = MinMaxScaler()
        return scaler.fit(np.array(action_samples).reshape(-1, 1))
    
    def normalize(self, actions):
        return self.scaler.transform(np.array(actions).reshape(-1, 1)).flatten()


# In[ ]:


def test(env, pid, x_force, y_force):    
    for episode in range(1, NUM_EPISODES + 1):
        old_state = None
        done = False
        total_reward = 0
        
        if SAVE_EACH_EPS_LOG:
            xpos, ypos, xvel, yvel, lander_angle, angular_vel = ([] for _ in range(6))
            rem_fuel, lander_mass, xpos_rocket, ypos_rocket, xpos_landingPad, ypos_landingPad = ([] for _ in range(6))
            fE, fS, pSi = ([] for _ in range(3))
            fE_pid = [] # temporary array to store PID generated Fe (which is not normalized yet)

            angle_deg, angvel_deg, psi_deg = ([] for _ in range(3))

            shaping_pos, shaping_vel, shaping_ang, shaping_angvel = ([] for _ in range(4))
            shaping_leftleg, shaping_rightleg, shaping_fe, shaping_fs = ([] for _ in range(4))
        
        if SAVE_SUMMARY:
            rew, total_rew = [], []
            ep, tot_step = [], []
            lastX, lastY, lastVx, lastVy, lastAngle, lastFuel = ([] for _ in range(6))

        if SIMULATE_WIND:
            wind_x, wind_y = ([] for _ in range(2))

        state = env.reset()
        max_steps = 500
        
        for t in range(max_steps): # env.spec.max_episode_steps
            old_state = state
            
            current_state = env.get_state_with_barge_and_landing_coordinates(untransformed_state=True)
            
            if SAVE_EACH_EPS_LOG:
                xpos.append(current_state[0]-current_state[12]) #xpos_rocket - xpos_landingPad
                ypos.append(current_state[1]-current_state[13]-0.520675) #ypos_rocket - ypos_landingPad
                xvel.append(current_state[2]) #xdot
                yvel.append(current_state[3]) #ydot
                lander_angle.append(current_state[4]) #theta
                angular_vel.append(current_state[5]) #theta_dot
                angle_deg.append(np.degrees(current_state[4]))
                angvel_deg.append(np.degrees(current_state[5]))
                rem_fuel.append(current_state[6]) #initial fuel = 0.2 * initial_mass
                lander_mass.append(current_state[7]) #initial_mass = 25.222
                xpos_rocket.append(current_state[0]) # xpos_rocket
                ypos_rocket.append(current_state[1]) # ypos_rocket
                xpos_landingPad.append(current_state[12]) # xpos_landingPad
                ypos_landingPad.append(current_state[13]) # ypos_landingPad
                
            action = list(pid.pid_algorithm(state)) #turn action into list so it can be modified
            if TURN_OFF_NOZZLE:
                action[2] = 0 # Set the nozzle angle to 0
            
            if SAVE_EACH_EPS_LOG: # append actions to array of actions
                fE_pid.append(action[0]) 
                fS.append(action[1])
                pSi.append(betterScale(action[2], -15*DEGTORAD, 15*DEGTORAD, -15*DEGTORAD, 15*DEGTORAD))
                psi_deg.append(betterScale(np.degrees(action[2]), -15, 15, -15, 15))
            
            # pass the state to the algorithm, get the actions    
            state, reward, shaping_element, done, _ = env.step(action)

            total_reward += reward
            
            if SAVE_EACH_EPS_LOG:
                rew.append(reward)
                shaping_pos.append(shaping_element[0])
                shaping_vel.append(shaping_element[1])
                shaping_ang.append(shaping_element[2])
                shaping_angvel.append(shaping_element[3])
                shaping_leftleg.append(shaping_element[4])
                shaping_rightleg.append(shaping_element[5])
                shaping_fe.append(shaping_element[6])
                shaping_fs.append(shaping_element[7])

            if SIMULATE_WIND:
                if state[LEFT_GROUND_CONTACT] == 0 and state[RIGHT_GROUND_CONTACT] == 0:
                    if t <= total_windstep:
                        env.apply_random_x_disturbance(epsilon=2, left_or_right=wind_dir, x_force=x_force)
                        if SAVE_EACH_EPS_LOG:
                            winds = env.get_winds_value()
                            wind_x.append(winds[0])
                            wind_y.append(winds[1])
            
            if done: # Go to next episode if done is True
                break # break out of the for loop
            
        if SAVE_EACH_EPS_LOG:
            save_dir = f"{model_dir}/logs/{test_name}_{NUM_EPISODES}Episodes/{test_name}_eps-{episode}_{round(total_reward, 2)}/"
            os.makedirs(save_dir, exist_ok=True)
            state_data=pd.DataFrame(list(zip(xpos,
                                            ypos,
                                            xvel,
                                            yvel,
                                            lander_angle,
                                            angular_vel,
                                            angle_deg,
                                            angvel_deg,
                                            rem_fuel,
                                            lander_mass,
                                            xpos_rocket,
                                            ypos_rocket,
                                            xpos_landingPad,
                                            ypos_landingPad)),
                                    columns=['x_pos',
                                            'y_pos',
                                            'x_vel',
                                            'y_vel',
                                            'lateral_angle',
                                            'angular_velocity',
                                            'angle_deg',
                                            'angvel_deg',
                                            'remaining_fuel',
                                            'lander_mass',
                                            'xpos_rocket',
                                            'ypos_rocket',
                                            'xpos_landingPad',
                                            'ypos_landingPad'])
    
            #find the step where the rocket has landed
            #this is used as the starting index to drop redundant rows from the dataframe
            try:
                landed_step = state_data.index[(state_data['y_vel']>-0.3) & (state_data['y_vel']<0.3)
                                            & (state_data['y_pos']<=0.3)].tolist()[0]
                print(f"Episode:\t{episode}\tReward:\t{round(total_reward,2)}\tSteps:\t{landed_step}")
            except:
                landed_step = None
                print(f"Episode:\t{episode}\tRocket doesn't land")
    
            #drop redundant rows from the state dataframe
            if landed_step is not None:
                state_data.drop(state_data.index[landed_step:], inplace = True)
        
            # Normalize PID generated Fe array datas and append them to the main Fe array
            try:
                fe_scaler = Scaler()
                fe_scaler.create_scaler(action_samples=fE_pid)
                fE.extend(fe_scaler.normalize(fE_pid))
            except:
                pass
    
            action_data=pd.DataFrame(list(zip(fE,fS,pSi,psi_deg)),columns=['Fe','Fs','Psi','Psi_deg'])
            #drop redundant rows from the action dataframe
            if landed_step is not None:
                action_data.drop(action_data.index[landed_step:], inplace = True)
        
            if SIMULATE_WIND:
                wind_dat= pd.DataFrame(list(zip(wind_x, wind_y)),columns=['x_wind force', 'y_wind force'])
                
            with pd.ExcelWriter(save_dir + f"{test_name}_{round(total_reward, 2)}.xlsx") as writer:
                state_data.to_excel(writer, sheet_name="state")
                action_data.to_excel(writer, sheet_name="action")
                if SIMULATE_WIND:
                    wind_dat.to_excel(writer, sheet_name="winds")    
            
            reward_data=pd.DataFrame(list(zip(rew,
                                            shaping_pos,
                                            shaping_vel,
                                            shaping_ang,
                                            shaping_angvel,
                                            shaping_leftleg,
                                            shaping_rightleg,
                                            shaping_fe,
                                            shaping_fs)),
                                    columns=['reward',
                                            'shaping_pos',
                                            'shaping_vel',
                                            'shaping_ang',
                                            'shaping_angvel',
                                            'shaping_leftleg',
                                            'shaping_rightleg',
                                            'r_fe',
                                            'r_fs'])
    
            #drop redundant rows from the dataframe
            if landed_step is not None:
                reward_data.drop(reward_data.index[landed_step:], inplace = True)
            
            with pd.ExcelWriter(save_dir + f"test-reward-elements_{test_name}_{round(total_reward, 2)}_episode{episode}.xlsx") as writer:
                reward_data.to_excel(writer, sheet_name=f"PID_test-reward-elements")
                
            print(f">>> Episode {episode} Logs saved")
    
            if SAVE_SUMMARY:            
                total_rew.append(reward_data['reward'].sum())
                ep.append(episode)
                tot_step.append(landed_step)
                
                #save the last state of the rocket for each episode
                lastX.append(state_data.at[state_data.index[-1], 'x_pos'])
                lastY.append(state_data.at[state_data.index[-1], 'y_pos'])
                lastVx.append(state_data.at[state_data.index[-1], 'x_vel'])
                lastVy.append(state_data.at[state_data.index[-1], 'y_vel'])
                lastAngle.append(state_data.at[state_data.index[-1], 'angle_deg'])
                lastFuel.append(state_data.at[state_data.index[-1], 'remaining_fuel'])
            
                summary_data=pd.DataFrame(list(zip(ep,tot_step,lastX,lastY,lastVx,lastVy,lastAngle,lastFuel,total_rew)),
                                        columns=['episode','total_step','lastX','lastY','lastVx','lastVy','lastAngle','lastFuel','total_reward'])
            
                summary_filename = model_dir + f"/logs/{test_name}_{NUM_EPISODES}Episodes" + f"/summary_{test_name}_{NUM_EPISODES}Episodes.xlsx"
                summary_sheetname = f"PID_test-summary"
                export_excel_append(summary_data, summary_filename, summary_sheetname)
            
            if SAVE_PLOT:    
                plot('state', state_data, save_dir, episode, total_reward)
                plot('action', action_data, save_dir, episode, total_reward)
    
        env.close()


# # Imports

# In[ ]:


import os
import numpy as np
from numpy.core.numeric import False_
import pandas as pd
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
import re

from control_and_ai.pid import PID_Benchmark

from constants import *
from constants import DEGTORAD
from environments.rocketlander_test_ga_constwind import RocketLander

simulation_settings = {'Side Engines': True,
                       'Clouds': True,
                       'Vectorized Nozzle': True,
                       'Graph': False,
                       'Render': True,
                       'Starting Y-Pos Constant': 1,
                       'Initial Force': 'random',
                       'Rows': 1,
                       'Columns': 2,
                       'Episodes': 1,
                       'Genes': genes}

env = RocketLander(simulation_settings)
#env = wrappers.Monitor(env, './rocket_videos', force=True)

model_dir = 'control_and_ai/PID'

#Initialize PID algorithm
pid = PID_Benchmark()


# # Main

# In[ ]:


print('Model:', 'PID_Benchmark')
print('Test Name:',test_name)
test(env, pid, x_force, y_force)

