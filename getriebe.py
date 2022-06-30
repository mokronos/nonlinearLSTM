import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def HD_friction_compliance(x,t):

    # i_m is the system input u, here only step function
    i_m = 0.28          # [A] 0.28 <= i_m <= 0.6

    # in the beginning the load is 0, try later on with positive load
    T_load = 0          # [Nm] < 0.6

    # friction an inertia of system outside of gear
    J_in = 5.2e-6		# [kg m^2]
    J_out = 0.0088		# [kg m^2]
    b_in = 1e-7			# [Nm/(deg/s)]
    b_out = 0.0005		# [Nm/(deg/s)]
    K_t = 0.0406		# [Nm/A]
    N = 80.0			# [-]

    # fiction parameters
    b_0 = 0.2894 *2.5	# [Nm]
    b_1 = 0.0005 *3.2	# [Nm/(deg/s)]
    b_2 = -2.8e-10 *3.2	# [Nm/(deg/s)^3]
    b_cyc = 0.0178 *1.8	# [Nm]
    phi_cyc = np.pi		# [rad]

    # spring constant
    k_1 = 5.594			# [Nm/deg]
    k_2 = 10.923		# [Nm/deg^3]


    # convert degree to rad
    p_conv = np.pi/180.0

    b_in = b_in/p_conv
    b_out = b_out/p_conv

    b_1 = b_1/p_conv
    b_2 = b_2/p_conv**3.0

    k_1 = k_1/p_conv
    k_2 = k_2/p_conv**3.0

    # asign physical names to states 
    theta_wg = x[0]         # [rad]
    theta_fs = x[1]         # [rad]
    d_theta_wg = x[2]       # [rad/s] 
    d_theta_fs = x[3]       # [rad/s]
    f = [0,0,0,0]

    # friction torque
    T_b = b_0 - b_1*d_theta_fs - b_2*d_theta_fs**3.0 + b_cyc*np.sin(theta_fs+phi_cyc)
    # ramp friction for the sake of continuity
    #if t < 0.05:
    #	T_b = ((t - 0.05)/0.05 + 1)*T_b

    # compliance torque
    T_k = k_1*(theta_wg-(-N*theta_fs)) + k_2*(theta_wg-(-N*theta_fs))**3.0

    # assign torque at wave generator and flex spline
    T_wg = T_k
    T_fs = N*T_k - T_b

    # differential equations
    f[0] = d_theta_wg
    f[1] = d_theta_fs
    f[2] = 1.0/J_in*(K_t*i_m - b_in*d_theta_wg - T_wg);
    f[3] = 1.0/J_out*(-b_out*d_theta_fs - T_fs + T_load);

    return f


t = np.arange(0, 5, 0.001)

x = odeint(HD_friction_compliance, [0,0,0,0],t)


fig, axs = plt.subplots(2)
axs[0].plot(t,x[:,2]*180.0/np.pi)
axs[1].plot(t,x[:,0]*180.0/np.pi)
plt.show()
