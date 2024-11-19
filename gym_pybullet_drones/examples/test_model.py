"""Sript to run a trained model n times , collect data and print statsitcs

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from stable_baselines3 import PPO

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = True
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 1
DEFAULT_MA = False
#Analysis varibales 
trials_num=10
filename = '/home/riotu/gym-pybullet-drones/gym_pybullet_drones/examples/results/NAMEOFMODEL-11.19.2024_13.51.42'
trial_data=[]
TARGET_POS = [1.0, 1.0, 1.0]  # Adjust to your actual target position
SUCCESS_THRESHOLD = 0.1  # Define a threshold for success

def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True, num_trials=1):
    
    
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path)
    test_env = HoverAviary(gui=gui,
                               obs=DEFAULT_OBS,
                               act=DEFAULT_ACT,
                               record=record_video)
    test_env_nogui = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)

    #### Loop for multiple trials
    for trial in range(trials_num):
        print(f"Starting trial {trial + 1}/{trials_num}")

        logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                        num_drones=DEFAULT_AGENTS if multiagent else 1,
                        output_folder=output_folder,
                        colab=colab
                        )

        mean_reward, std_reward = evaluate_policy(model,
                                                  test_env_nogui,
                                                  n_eval_episodes=10
                                                  )
        
        print(f"\n\n\nTrial {trial + 1} - Mean reward: {mean_reward} +- {std_reward}\n\n")
        
        obs, info = test_env.reset(options={})
        start = time.time()
        for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
            action, _states = model.predict(obs,
                                            deterministic=True
                                            )
            obs, reward, terminated, truncated, info = test_env.step(action)
            trial_data.append({
                "trial_number": trial + 1,
                "step": i,
                "X": obs[0][0],   # X Position
                "Y": obs[0][1],   # Y Position
                "Z": obs[0][2],   # Z Position
                "Q1": obs[0][3],  # Quaternion Q1
                "Q2": obs[0][4],  # Quaternion Q2
                "Q3": obs[0][5],  # Quaternion Q3
                "Q4": obs[0][6],  # Quaternion Q4
                "R": obs[0][7],   # Roll
                "P": obs[0][8],   # Pitch
                "Y": obs[0][9],   # Yaw
                "VX": obs[0][10], # Linear Velocity X
                "VY": obs[0][11], # Linear Velocity Y
                "VZ": obs[0][12], # Linear Velocity Z
                "WX": obs[0][13], # Angular Velocity X
                "WY": obs[0][14], # Angular Velocity Y
                "WZ": obs[0][15], # Angular Velocity Z
                "RPM1": obs[0][-4], # Motor RPM1
                "RPM2": obs[0][-3], # Motor RPM2
                "RPM3": obs[0][-2], # Motor RPM3
                "RPM4": obs[0][-1], # Motor RPM4
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
                "filename": filename
                })

            obs2 = obs.squeeze()
            act2 = action.squeeze()
            print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated,"TargetPose",test_env.TARGET_POS)
            if DEFAULT_OBS == ObservationType.KIN:
             
                logger.log(drone=0,
                           timestamp=i/test_env.CTRL_FREQ,
                           state=np.hstack([obs2[0:3],
                                            np.zeros(4),
                                            obs2[3:15],
                                            act2
                                                ]),
                               control=np.zeros(12)
                            )
        
            
            if terminated or truncated:
                #obs, info = test_env.reset(seed=42, options={})
                #continue
                break

            test_env.render()
            sync(i, start, test_env.CTRL_TIMESTEP)

        #if plot and DEFAULT_OBS == ObservationType.KIN:
        #    logger.plot()    
    test_env.close()

    data_pd = pd.DataFrame(trial_data)  
    output_file = "step_data_1" + filename.split('/')[-1] + ".csv"
    data_pd.to_csv(output_file, index=False) 
      
    print("statistics")
    print("All trials completed.")

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))