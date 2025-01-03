"""Sript to run a trained model 

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
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



def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True, num_trials=1):
    
    filename = '/home/riotu/gym-pybullet-drones/gym_pybullet_drones/examples/results/save-11.05.2024_13.00.38'

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
    num_trials=4
    for trial in range(num_trials):
        print(f"Starting trial {trial + 1}/{num_trials}")

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
        
        obs, info = test_env.reset(seed=42, options={})
        start = time.time()
        for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
            action, _states = model.predict(obs,
                                            deterministic=True
                                            )
            obs, reward, terminated, truncated, info = test_env.step(action)
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
        
            test_env.render()
            sync(i, start, test_env.CTRL_TIMESTEP)
            if terminated:
                obs = test_env.reset(seed=42, options={})
        if plot and DEFAULT_OBS == ObservationType.KIN:
            logger.plot()    
    test_env.close()

        

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