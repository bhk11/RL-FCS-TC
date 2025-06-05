import numpy as np


def customized_reward(env, observation, action, safeguardActive, current_total_SG):



        
    i_d = observation[1] # d-axis current
    i_q = observation[2] # q-axis current
    u_d = observation[3] # d-axis voltage
    u_q = observation[4] # q-axis voltage
    cos_eps = observation[5]  # cosine of the electrical angle
    sin_eps = observation[6]  # sine of the electrical angle
    current_total = observation[7]  # stator current i_s
    T_ref = observation[8]  # reference torque
    #get the torque information from the env, because the torque shaft should be used only during training and, thus, the torque is not part of the observation
    T = env.get_wrapper_attr('physical_system').system_state[env.get_wrapper_attr('state_names').index('torque')]

    dangerzone_boundary = 240/270
    id_boundary = id_boundary = 15 / 270
    torque_boundary = 5 / 200

    e_T_abs = np.abs(T_ref - T)
    gamma = 0.868   # discount factor WARNING -> hardcoded
    term = False

    if current_total > 1: # region E, "Error zone", set the terminal flag
            rew = -1*current_total
            term = True
            #print(current_total)
        
    if (current_total_SG > 1) & (safeguardActive): # region E, "Error zone", set the terminal flag
            rew = -(1-gamma)

    elif current_total > dangerzone_boundary: # region D, "Danger zone", short time overcurrent
            reward_offset = - (1 - gamma)
            rew = (1 - (current_total - dangerzone_boundary) / (1 - dangerzone_boundary)) * (1 - gamma) / 2 + reward_offset

    elif (current_total_SG > dangerzone_boundary) & (safeguardActive): # region D_S, "Danger zone", short time overcurrent
            rew = - (1 - gamma)/2
            
    elif i_d > id_boundary: # region C, "Caturation zone", saturation of the permanent magnet
            reward_offset = - (1 - gamma) / 2
            rew = (1 - (i_d - id_boundary) / (dangerzone_boundary - id_boundary)) * (1 - gamma) / 2 + reward_offset

    elif (e_T_abs > torque_boundary) & (safeguardActive): # region B_S, "Basic zone", torque is not yet accurate
            rew = 0

    elif e_T_abs > torque_boundary: # region B, "Basic zone", torque is not yet accurate
            reward_offset = 0
            rew = (1 - e_T_abs / 2) * (1 - gamma) / 2 + reward_offset #+ rew_6s
        
    else: # region A, "Awesome zone", torque is accurate and current needs to be minimized
            reward_offset = (1 - gamma) / 2
            rew = (1 - current_total) * (1 - gamma) / 2 + reward_offset


    return rew, term