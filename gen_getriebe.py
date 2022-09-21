import numpy as np
from helper import  gen_data, save_dataset
from odes.getriebe import HD_friction_compliance



config = {
        "name":"getriebe_simple",
        "function": "HD_friction_compliance",
        "samples": 1000,
        "time": "t",
        # theta_wg, theta_fs, d_theta_wg, d_theta_fs
        "outputs": ["d_rot_out", "d_rot_in", "rot_out", "rot_in"],
        "init": [[0,0,0,0]],
        "timestep": 0.01,
        "constants": {
                },
        "inputs": {
                # needs to be be 0.28 <= u <=0.6
                "i_m":{
                    "types": {
                        "steps": {
                            "when":
                            [[0],
                             [0],
                             [0],
                             [0,0.2,0.4]],
                            "height":
                            [[0.28],
                             [0.5],
                             [0.6],
                             [0.28,0.5,0.6]]
                            },
                        }},
        }} 



data = gen_data(config, HD_friction_compliance)
save_dataset(data, config)
