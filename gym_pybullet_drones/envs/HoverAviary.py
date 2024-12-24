import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class HoverAviary(BaseRLAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.TARGET_POS = np.array([3,2,1])
        self.EPISODE_LEN_SEC = 20
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        #state is [ 6.01253783e-03  7.09888725e-04  1.11080019e-01 -4.60590213e-02
        #1.06515784e-01  3.10667150e-02  9.92757681e-01 -8.69609693e-02
        #2.16026856e-01  5.31307556e-02  1.26284143e-01  3.15567905e-02
        #4.64238211e-03 -1.67039733e+00  2.25731599e+00  1.13286253e-01
        #1.51918496e+04  1.51757607e+04  1.51918496e+04  1.41037490e+04]

        state = self._getDroneStateVector(0)
        #print("state is",state)
        #ret=-0.1*(abs(np.linalg.norm(self.TARGET_POS-state[0:3])))
        #ret = max(0, 5 - np.linalg.norm(self.TARGET_POS-state[0:3])**4)

        #Reward structure 
        # reward for getting closer and penality for being away certain threshold 
        distance_to_goal=np.linalg.norm(self.TARGET_POS - state[0:3])
        distance_reward=np.exp(-0.5*distance_to_goal)
        # Distance penalty 
        
        distance_penalty = -0.1 * abs(np.linalg.norm(self.TARGET_POS - state[0:3]))
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < 0.5:
           distance_penalty=distance_penalty+1500
        #sperating distance of xy from z
        x_y_penalty = -0.1 * abs(np.linalg.norm(self.TARGET_POS[0:2] - state[0:2]))
        z_penalty = -0.1 * abs(self.TARGET_POS[2] - state[2])
        if np.linalg.norm(self.TARGET_POS[0:2]-state[0:2]) < 0.5:
           x_y_penalty=x_y_penalty+15
        #altitidue reward
        altitude_reward=1.0 if abs(self.TARGET_POS[2]-state[2])<0.1 else 0
        # Speed penalty to limit linear velocity
        speed_penalty = -0.05 * np.linalg.norm(state[10:13])  # VX, VY, VZ

        # Angular velocity penalty to minimize rapid changes in direction
        angular_velocity_penalty = -0.05 * np.linalg.norm(state[13:16])  # WX, WY, WZ

        # Distance-based speed penalty to slow down as the drone gets closer to the target
        distance_based_speed_penalty = -0.1 * np.linalg.norm(state[10:13]) / (0.3 + np.linalg.norm(self.TARGET_POS - state[0:3]))
        # Distance penalty 
        expo_distance_penalty = -0.1 * np.exp(abs(np.linalg.norm(self.TARGET_POS - state[0:3])))
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < 0.5:
           expo_distance_penalty=expo_distance_penalty+15
        # Total reward
        ret = angular_velocity_penalty + distance_based_speed_penalty+distance_reward
        #+ speed_penalty + angular_velocity_penalty + distance_based_speed_penalty

        return ret

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < 0.5:
            return True
        else:
            return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if abs(state[0])>10.0 or abs(state[1])>10.0 or state[2]>2.0:
        #or abs(state[7]) > .4 or abs(state[8]) > .4): # Truncate when the drone is too tilted
       
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
