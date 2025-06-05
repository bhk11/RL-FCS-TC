import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

from src.agent.SafeGuard import SafeGuard
from src.agent.customReward import customized_reward

# Modify Observation space to get flat vector instead of tuple
# that's why it is customized here
class FeatureWrapper(ObservationWrapper):
    """
    Wrapper class which wraps the environment to change its observation. Serves
    the purpose to improve the agent's learning speed.
    
    It changes epsilon to cos(epsilon) and sin(epsilon). This serves the purpose
    to have the angles -pi and pi close to each other numerically without losing
    any information on the angle.
    
    Additionally, this wrapper adds a new observation i_sd**2 + i_sq**2. This should
    help the agent to easier detect incoming limit violations.
    """

    def __init__(self, env):
        """
        Changes the observation space to fit the new features
        
        Args:
            env(GEM env): GEM environment to wrap
        """
        super(FeatureWrapper, self).__init__(env)
        self.OMEGA_IDX = env.get_wrapper_attr('state_names').index('omega')
        self.I_SQ_IDX = env.get_wrapper_attr('state_names').index('i_sq')
        self.I_SD_IDX = env.get_wrapper_attr('state_names').index('i_sd')
        self.U_SQ_IDX = env.get_wrapper_attr('state_names').index('u_sq')
        self.U_SD_IDX = env.get_wrapper_attr('state_names').index('u_sd')
        self.EPSILON_IDX = env.get_wrapper_attr('state_names').index('epsilon')

        self.angle_weighting = 0.1  # Weighting factor for cos and sin values

        new_low = np.concatenate(([self.env.observation_space.low[self.OMEGA_IDX]], # angular velocity
                            self.env.observation_space.low[self.I_SD_IDX:self.I_SQ_IDX+1], # currents in dq
                            self.env.observation_space.low[self.U_SD_IDX:self.U_SQ_IDX+1], # voltages in dq
                            [-1, -1], # angles in cos, sin 
                            [-1],    # stator current
                            [-1]))#,   # torque reference
                            #[-1, -1, -1])) # active action -> action is added in last action wrapper
        new_high = np.concatenate(([self.env.observation_space.high[self.OMEGA_IDX]],
                            self.env.observation_space.high[self.I_SD_IDX:self.I_SQ_IDX+1],
                            self.env.observation_space.high[self.U_SD_IDX:self.U_SQ_IDX+1],
                            [+1, +1],
                            [+1],
                            [+1]))#,
                            #[+1, +1, +1]))
        self.observation_space = Box(new_low, new_high)

    def observation(self, observation):
        """
        Gets called at each return of an observation. Adds the new features to the
        observation and removes original epsilon.
        
        """
        cos_eps = self.angle_weighting * np.cos(observation[self.EPSILON_IDX])
        sin_eps = self.angle_weighting * np.sin(observation[self.EPSILON_IDX])
        current_vec_nom = np.sqrt(observation[self.I_SQ_IDX]**2 + observation[self.I_SD_IDX]**2)  
        if observation[self.OMEGA_IDX] > 1:
            print('something is wrong')
        observation = np.concatenate(([observation[self.OMEGA_IDX]],
                                    observation[self.I_SD_IDX:self.I_SQ_IDX + 1],
                                    observation[self.U_SD_IDX:self.U_SQ_IDX + 1],
                                    np.array([cos_eps, sin_eps]),
                                    np.array([current_vec_nom]),
                                    np.array([self.env.get_wrapper_attr('reference_generator')._reference_value])
                                    ))
        return observation
    

# Wrapper to add last action to the observation space and modify sin and cos values
class LastActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super(LastActionWrapper, self).__init__(env)
        state_space = self.env.observation_space

        self.subactions = -np.power(-1, self.env.get_wrapper_attr('physical_system')._converter._subactions)
    

        new_low = np.concatenate((state_space.low,
                                  self.subactions[0]))
        new_high = np.concatenate((state_space.high,
                                   self.subactions[-1]))



        self.observation_space = Box(new_low, new_high)

        self.SafeGuard = SafeGuard(self.env)

    # fill last action with zeros
    def reset(self, seed = 42, **kwargs):
        super().reset(seed = seed)
        observation, info = self.env.reset(**kwargs)
        self.last_observation = observation
        self.last_action = self.subactions[0]
        self.last_reward = np.nan
        return np.concatenate((observation, self.last_action)), info

    # put action to the last action
    # added safe guarding and individual reward function here.
    def step(self, action):

        #safe guarding
        action, safeguardActive, current_total_SG = self.SafeGuard.check_action(action, self.last_action, self.last_observation)

        observation, reward, terminated, truncated, info = self.env.step(action)

        reward, term = customized_reward(self, observation, action, safeguardActive, current_total_SG)

        self.last_observation = observation
        self.last_action = self.subactions[action]
        self.last_reward = reward

        return np.concatenate((observation, self.last_action)), reward, terminated, truncated, info