import numpy as np
from helper import  gen_data, save_dataset


def thermal(x, t, u):
    
    R_12 = 0.035
    R_23 = 0.02
    C_1 = 5
    C_2 = 45
    C_3 = 40
    Volumeflow = u[3]
    Tinlet = u[4]

    R_23=R_23*(8/Volumeflow)**(0.4-0.0434*(1-(22.65)/(Tinlet)))*((22.65)/(Tinlet))**(0.146)
    Rfluid=1/(Volumeflow/60000*3.31e3*1059) # 1/(Vdot*cp*rho)

    dxdt =[u[0]/C_1 - x[0]/(C_1*R_12) + x[1]/(C_1*R_12),
    x[0]/(C_2*R_12) + x[2]/(C_2*R_23) - (x[1]*(1/R_12 + 1/R_23))/C_2,
    x[1]/(C_3*R_23) - (x[2]*(1/R_23 + 1/Rfluid))/C_3 + (Tinlet/Rfluid)/C_3,
    u[1]/C_1 - x[3]/(C_1*R_12) + x[4]/(C_1*R_12),
    x[3]/(C_2*R_12) + x[5]/(C_2*R_23) - (x[4]*(1/R_12 + 1/R_23))/C_2,
    x[4]/(C_3*R_23) - (x[5]*(1/R_23 + 1/Rfluid))/C_3 + (x[2]/Rfluid)/C_3,
    u[2]/C_1 - x[6]/(C_1*R_12) + x[7]/(C_1*R_12),
    x[6]/(C_2*R_12) + x[8]/(C_2*R_23) - (x[7]*(1/R_12 + 1/R_23))/C_2,
    x[7]/(C_3*R_23) - (x[8]*(1/R_23 + 1/Rfluid))/C_3 + (x[5]/Rfluid)/C_3]

    return dxdt
    

config = {
        "name":"thermal_simple",
        "function": "thermal",
        "samples": 1000,
        "time": "t",
        "outputs": ["Chip U", "Sink U","Fluid U","Chip V", "Sink V","Fluid V","Chip W", "Sink W","Fluid W"],
        "init": [[20,20,20,20,20,20,20,20,20]],
        "timestep": 0.01,
        "constants": {
                },
        "inputs": {
                "Cur_U":{
                    "types": {
                        "steps": {
                            "when":
                            [[0],
                             [0,0.2,0.4,0.6,0.8],
                             [0,0.2,0.4,0.6,0.8]],
                            "height":
                            [[1000],
                             [0,1000,2000,3000,4000],
                             [0,1000,500,2000,1500]],
                            },
                        }},
                "Cur_V":{
                    "types": {
                        "steps": {
                            "when":
                            [[0],
                             [0,0.2,0.4,0.6,0.8],
                             [0,0.2,0.4,0.6,0.8]],
                            "height":
                            [[1000],
                             [0,1000,2000,3000,4000],
                             [0,1000,500,2000,1500]],
                            },
                        }},
                "Cur_W":{
                    "types": {
                        "steps": {
                            "when":
                            [[0],
                             [0,0.2,0.4,0.6,0.8],
                             [0,0.2,0.4,0.6,0.8]],
                            "height":
                            [[1000],
                             [0,1000,2000,3000,4000],
                             [0,1000,500,2000,1500]],
                            },
                        }},
                "Volumeflow":{
                    "types": {
                        "steps": {
                            "when":
                            [[0],
                             [0, 0.5]],
                            "height":
                            [[8],
                             [8, 10]],
                            },
                        }},
                "Tinlet":{
                    "types": {
                        "steps": {
                            "when":
                            [[0],
                             [0, 0.5]],
                            "height":
                            [[20],
                             [20, 30]],
                            },
                        }},
        }} 



data = gen_data(config, thermal)
save_dataset(data, config)
