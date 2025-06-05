import numpy as np
import random

class SafeGuard:
    """
    SafeGuard is a class that provides safety checks for the agent's actions.
    It ensures that the agent does not perform any harmful or unsafe actions.
    """

    def __init__(self, env):
        
        #sample time
        self.tau = env.get_wrapper_attr('physical_system').tau

        # motor parameters
        self.p = env.get_wrapper_attr('physical_system').electrical_motor._motor_parameter.get('p')
        self.r_s = env.get_wrapper_attr('physical_system').electrical_motor._motor_parameter.get('r_s')
        self.l_d = env.get_wrapper_attr('physical_system').electrical_motor._motor_parameter.get('l_d')
        self.l_q = env.get_wrapper_attr('physical_system').electrical_motor._motor_parameter.get('l_q')
        self.psi_p = env.get_wrapper_attr('physical_system').electrical_motor._motor_parameter.get('psi_p')
        self.tau_d = self.l_d / self.r_s
        self.tau_q = self.l_q / self.r_s
        
        # limit values for the normalization
        self.i_lim = env.get_wrapper_attr('physical_system').limits[env.get_wrapper_attr('state_names').index('i_sd')]
        self.u_lim = env.get_wrapper_attr('physical_system').limits[env.get_wrapper_attr('state_names').index('u_sd')]
        self.omega_lim = env.get_wrapper_attr('physical_system').limits[env.get_wrapper_attr('state_names').index('omega')]

        self.u_dc = 2 * self.u_lim
        self.subactions = -np.power(-1, env.get_wrapper_attr('physical_system')._converter._subactions)
        self.abc_to_dq = env.get_wrapper_attr('physical_system').abc_to_dq_space

        # define the limit values for the current and voltage
        self.limitCurrent = 240                         # A - Boundary from Region C to D in the reward function
        self.limitVoltage = 2/(np.pi) * self.u_dc       # V - Limit of the controlability of the system 

        # define the constant input matrix of the system model
        self.B = np.array([[1 / self.l_d,            0],
                           [           0, 1 / self.l_q]])


    def check_action(self, action, last_action, observation):
        """
        Check if the action is safe to perform.

        :param action: The action to check.
        :return: safe_action: The action if it is safe, otherwise a safe alternative action.
        """
        w_me = observation[0] * self.omega_lim  # mechanical angular velocity
        w_el = w_me * self.p  # electrical angular velocity
        i_d = observation[1] * self.i_lim  # d-axis current
        i_q = observation[2] * self.i_lim  # q-axis current
        u_d = observation[3] * self.u_lim  # d-axis voltage
        u_q = observation[4] * self.u_lim # q-axis voltage
        cos_eps = observation[5]  # cosine of the electrical angle
        sin_eps = observation[6]  # sine of the electrical angle
        




        eps = np.arctan2(sin_eps, cos_eps)

        # define the non-constant dynamic matrix A and disturbance matrix E
        A = np.array([[            - 1 / self.tau_d,  w_el * self.l_q / self.l_d],
                      [- w_el * self.l_d / self.l_q,            - 1 / self.tau_q]])

        E = np.array([[                             0],
                      [- w_el * self.psi_p / self.l_q]])

        i_dq_k = np.array([[i_d],
                          [i_q]])


        # predict the motor currents at the next sampling step, 
        # this needs to be done to incorporate the dead time of the plant in digitally controlled systems
        u_abc_k  = self.u_lim * last_action
        u_dq_k = np.transpose(np.array([self.abc_to_dq(u_abc_k, epsilon_el=eps + w_el * self.tau * 0.5)]))
        # note that "@" is the matrix multiplication operator in python
        i_dq_k1 = i_dq_k + self.tau * (A @ i_dq_k + 
                                       self.B @ u_dq_k + 
                                       E)
        active = False

        u_abc_k1  = self.u_lim * self.subactions[action]
        u_dq_k1 = np.transpose(np.array([self.abc_to_dq(u_abc_k1, epsilon_el=eps + w_el * self.tau * 0.5)]))

        # predict the motor currents at the next sampling step
        i_dq_k2 = i_dq_k1 + self.tau * (A @ i_dq_k1 + 
                                       self.B @ u_dq_k1 + 
                                       E)
        # predict the voltage for stationary operation
        u_dq_k2 = np.linalg.inv(self.B) @ ((np.identity(2)-A) @ i_dq_k2 - E)

        current_total_SG = np.linalg.norm(i_dq_k2) / self.i_lim
        if (np.linalg.norm(i_dq_k2) <= self.limitCurrent) & (np.linalg.norm(u_dq_k2) <= self.limitVoltage):
            # if the action is safe, return the action and active flag
            return action, active, current_total_SG 
        
        else:
            active = True
            # predict the currents and stationary voltage for all possible actions
            i_s_k2_all = []
            u_s_k2_all = []
            for a in range(len(self.subactions)):
                u_abc_k1  = self.u_lim * self.subactions[a]
                u_dq_k1 = np.transpose(np.array([self.abc_to_dq(u_abc_k1, epsilon_el=eps + w_el * self.tau * 0.5)]))
                i_dq_k2 = i_dq_k1 + self.tau * (A @ i_dq_k1 + 
                                                          self.B @ u_dq_k1 + 
                                                          E)
                i_s_k2_all.append(np.linalg.norm(i_dq_k2))
                u_s_k2_all.append(np.linalg.norm(np.linalg.inv(self.B) @ ((np.identity(2)-A) @ i_dq_k2 - E)))
                # check which actions are safe

            action_eval = (np.array(i_s_k2_all) <= self.limitCurrent) & (np.array(u_s_k2_all) <= self.limitVoltage)
            #if no action is safe, choose the action with the lowest current
            if sum(action_eval) == 0:
                action_eval[np.argmin(i_s_k2_all)] = True
            
            act_safe = np.where(action_eval == True)[0]
            save_action = act_safe[random.choice(range(0,len(act_safe)))]

            return save_action, active, current_total_SG