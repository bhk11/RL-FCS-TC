import numpy as np
from pathlib import Path

import gym_electric_motor as gem
from gym_electric_motor.reference_generators import ConstReferenceGenerator
from gym_electric_motor.physical_systems.mechanical_loads import ExternalSpeedLoad
from gym_electric_motor.physical_system_wrappers import CosSinProcessor, DeadTimeProcessor
from gym_electric_motor.envs.motors import ActionType, ControlType, Motor, MotorType
from gym_electric_motor.physical_systems.solvers import EulerSolver#ScipySolveIvpSolver

from gymnasium.wrappers import FlattenObservation, TimeLimit

from src.env.randomSpeed import randomSpeedProfile

from stable_baselines3 import DQN
from stable_baselines3.common.utils import get_linear_fn
from .customWrapper import FeatureWrapper, LastActionWrapper
from .customLearningRateSchedule import linear_schedule
from .CustomMlpPolicy import CustomMlpPolicy


def training(params_dict):
    """
    Traing the DQN agent with the given parameters.
    Args:
        params_dict (dict): Dictionary containing the parameters for training.
    """

    # Unpack parameters
    #"FCS_RL_agent",
    
    gamma = params_dict.get("gamma")
    
    #learning rate
    alpa0 = params_dict.get("alpha0")
    alpha1 = params_dict.get("alpha1")
    lr_reduction_start = params_dict.get("lr_reduction_start")
    lr_reduction_interval = params_dict.get("lr_reduction_interval")

    # archetecture
    layers = params_dict.get("layers")
    neurons = params_dict.get("neurons")
    epsilon0 = params_dict.get("epsilon0")
    epsilon1 = params_dict.get("epsilon1")
    nb_policy_annealing_steps = params_dict.get("nb_policy_annealing_steps")

    activation_fcn = params_dict.get("activation_fcn")
    activation_fcn_parameter = params_dict.get("activation_fcn_parameter")

    buffer_size = params_dict.get("memory_size")
    batch_size = params_dict.get("batch_size")
    target_update_interval = params_dict.get("target_update_parameter")

    nb_steps = params_dict.get("nb_training_steps")
    max_episode_steps = params_dict.get("nb_episode_steps")

    verbose = params_dict.get("verbose")
    


    # Define environment parameters
    envtype = Motor(
        MotorType.PermanentMagnetSynchronousMotor,
        ControlType.TorqueControl,
        ActionType.Finite
    )

    motor_parameter = dict(p=3,            # [p] = 1, nb of pole pairs
                       r_s=17.932e-3,  # [r_s] = Ohm, stator resistance
                       l_d=0.37e-3,    # [l_d] = H, d-axis inductance
                       l_q=1.2e-3,     # [l_q] = H, q-axis inductance
                       psi_p=65.65e-3, # [psi_p] = Vs, magnetic flux of the permanent magnet
                       )  # BRUSA

    u_sup = 350
    nominal_values=dict(omega=12000*2*np.pi/60,
                    i=240,
                    u=u_sup)

    limit_values=nominal_values.copy()
    limit_values["i"] = 270
    limit_values["torque"] = 200

    sampling_time = 50e-6

    pmsm_init= {
        'states': {
            'i_sd': 0.0,
            'i_sq': 0.0,
            'epsilon': 0.0,
        }
    }

    # define the physical system wrapper
    physical_system_wrappers = [
        # Wrapped directly around the physical system
        DeadTimeProcessor(steps=1) # Only use DeadTimeWrapper after you have implemented a last action concatinator for the state
        ]
    
    torque_ref_generator = ConstReferenceGenerator(reference_state='torque', reference_value=np.random.uniform(-1, 1))
    random_profile_generator = randomSpeedProfile(maxSpeed=nominal_values["omega"], 
                                                  epsLength=max_episode_steps)
    

    # Create the environment
    env = gem.make(envtype.env_id(),
                   motor = dict(
                       motor_parameter=motor_parameter,
                       limit_values=limit_values,
                       nominal_values=nominal_values,
                       motor_initializer=pmsm_init,
                   ),
                   supply=dict(u_nominal=u_sup),
                   physical_system_wrappers=physical_system_wrappers, # Pass the Physical System Wrappers
                   load=ExternalSpeedLoad(random_profile_generator.randomProfile, 
                                          tau=sampling_time),
                   tau=sampling_time,
                   #reward_function=WeightedSumOfErrors(reward_weights={'torque': 1},  # but the reward distribution will be overwritten
                  #                                            gamma=gamma), # by means of the defined wrapper function
                   reference_generator=torque_ref_generator,
                   ode_solver=EulerSolver()
                   )

    env = TimeLimit(LastActionWrapper(FeatureWrapper(FlattenObservation(env))), max_episode_steps = max_episode_steps)
    # applying wrappers



    policy_kwargs = dict(
        #observation_space = env.observation_space,
        #action_space = env.action_space,
        activation_fn = [('LeakyReLU', activation_fcn_parameter)]*layers + [None],
        net_arch =[neurons]*layers,                
                        )


    agent = DQN(CustomMlpPolicy, env, buffer_size=buffer_size, learning_starts=1000 ,train_freq=1, 
            batch_size=batch_size, gamma=gamma, policy_kwargs=policy_kwargs,
            learning_rate=linear_schedule(alpa0, alpha1, (lr_reduction_start/nb_steps), (lr_reduction_interval/nb_steps)),
            exploration_final_eps=epsilon1, exploration_initial_eps=epsilon0, target_update_interval=target_update_interval,
            verbose=verbose)
    agent.learn(total_timesteps=nb_steps)


    log_path = Path.cwd() / "saved_agents" 
    log_path.mkdir(parents=True, exist_ok=True)
    agent.save(str(log_path / "TutorialAgent"))