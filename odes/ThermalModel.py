from scipy.integrate import odeint
from numpy import zeros, ones, arange

from matplotlib import pyplot


class thermal_network:
    def __init__(self, R_12, R_23, C_1, C_2, C_3, Volumeflow, Tinlet, x_0):
        """Inits thermal network
        # Arguments
            R_12
            R_23
            C_1
            C_2
            C_3
            Volumeflow
            Tinlet
            """
        
        self.R_12=R_12
        self.R_23=R_23
        self.C_1=C_1
        self.C_2=C_2
        self.C_3=C_3
        self.Volumeflow=Volumeflow
        self.Tinlet=Tinlet
        self.x_0=x_0

    def ode(self, x, t, u):
        """Dynamic thermal network

        # Arguments
            x: [temperatures]
            t: Time steps for ode solving
            u: Heat flow

        # Returns
            Derivative of internal states
        """

        # ODE of thermal network

        R_23=self.R_23*(8/self.Volumeflow)**(0.4-0.0434*(1-(22.65)/(self.Tinlet)))*((22.65)/(self.Tinlet))**(0.146)
        Rfluid=1/(self.Volumeflow/60000*3.31e3*1059) # 1/(Vdot*cp*rho)

        dxdt =[u[0]/self.C_1 - x[0]/(self.C_1*self.R_12) + x[1]/(self.C_1*self.R_12),
        x[0]/(self.C_2*self.R_12) + x[2]/(self.C_2*R_23) - (x[1]*(1/self.R_12 + 1/R_23))/self.C_2,
        x[1]/(self.C_3*R_23) - (x[2]*(1/R_23 + 1/Rfluid))/self.C_3 + (self.Tinlet/Rfluid)/self.C_3,
        u[1]/self.C_1 - x[3]/(self.C_1*self.R_12) + x[4]/(self.C_1*self.R_12),
        x[3]/(self.C_2*self.R_12) + x[5]/(self.C_2*R_23) - (x[4]*(1/self.R_12 + 1/R_23))/self.C_2,
        x[4]/(self.C_3*R_23) - (x[5]*(1/R_23 + 1/Rfluid))/self.C_3 + (x[2]/Rfluid)/self.C_3,
        u[2]/self.C_1 - x[6]/(self.C_1*self.R_12) + x[7]/(self.C_1*self.R_12),
        x[6]/(self.C_2*self.R_12) + x[8]/(self.C_2*R_23) - (x[7]*(1/self.R_12 + 1/R_23))/self.C_2,
        x[7]/(self.C_3*R_23) - (x[8]*(1/R_23 + 1/Rfluid))/self.C_3 + (x[5]/Rfluid)/self.C_3]

        return dxdt


    def update(self, u, dt):
        """Integration of thermal ode

        # Arguments
            u: Heat flow und Zeitschrittweite

        # Returns
            Temperature
        """

        # Solving ODE with scipy library
        x = odeint(self.ode, self.x_0, [0,dt], args=(u,))

        self.x_0 = x[1] # Ergebnis der Zustaende nach dem Zeitschritt

        return x[1]


# Simulation of thermal network class
dt=0.01 # Zeitschrittweite
thermalNet = thermal_network(R_12=0.035, R_23=0.02, C_1=5, C_2=45, C_3=40, Volumeflow=8, Tinlet=20 ,x_0=ones(9)*20)
# change volume flow (use as input) and tinlet
SolT = list()
tvec=arange(0,10,dt) # Simulationszeit 10s und Zeitschrittweite dt
u=[1000,1000,1000] # Waermestrom auf die Chips der Phasen U,V und W.
# Volumeflow und Inlettemperatur können auch als Eingaenge in u integriert werden für Zeitabhaengigkeit
for t in range(int(10/dt)):
    SolT.append(thermalNet.update(u,dt)) # evtl. noch Zeitabhaengig machen u[t] zum Profile vorgeben

pyplot.plot(tvec,SolT)
pyplot.legend(['Chip U', 'Sink U','Fluid U','Chip V', 'Sink V','Fluid V','Chip W', 'Sink W','Fluid W',])
pyplot.xlabel('Time in s')
pyplot.ylabel('Temperature in °C')
pyplot.grid()
pyplot.show()
