import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import argparse

class ParticleSimulator(object):
    def __init__(self,
            init_pos=np.array([2.0, 0.0]),
            init_vel=np.array([0.0, 0.0]),
            MASS=np.array([0.5]),
            L=np.array([2.0]),
            G=np.array([0.0,-9.81]),
            K=np.array([6.0]),
            DAMPING=np.array([0.005]),
            output_vel=True,
            output_mass=False,
            output_l=False,
            output_g=False,
            output_damping=False,
            output_k=False):

        self.x=init_pos
        self.v=init_vel

        self.MASS=MASS
        self.L=L
        self.G=G 
        self.K=K
        self.DAMPING=DAMPING

        self.output_vel=output_vel
        self.output_mass=output_mass
        self.output_l=output_l
        self.output_g=output_g
        self.output_k=output_k
        self.output_damping=output_damping

    def reset(self,vel=np.array([0.0,0.0])):
        self.initialize_pos()
        self.v = vel

    def params_at(self,t):
        params = [self.pos()]
        if (self.output_vel):
            params+= [self.vel()]
        if (self.output_mass):
            params+= [self.mass(t)]
        if (self.output_l):
            params+= [self.l(t)]
        if (self.output_g):
            params+= [self.g(t)]
        if (self.output_k):
            params+= [self.k(t)]
        if (self.output_damping):
            params+= [self.damping(t)]
        return np.concatenate(params, axis=0)

    def pos_initializer(self):
        def init(self):
            return self.x
        return init
    
    def pos_initializer_is(self, func):
        self.pos_initializer = func

    def initialize_pos(self):
        self.x = self.pos_initializer()

    def pos(self):
        return self.x

    def pos_is(self,pos):
        self.x = pos

    def vel(self):
        return self.v

    def vel_is(self,vel):
        self.v = vel 

    def mass(self,t):
        return self.MASS 
    def l(self,t):
        return self.L 
    def g(self,t):
        return self.G 
    def k(self,t):
        return self.K 
    def damping(self,t):
        return self.DAMPING 

    def spring_f(self,x,v,t):
        norm_x = np.linalg.norm(x)
        unit_x = x/np.linalg.norm(x)
        return self.k(t)*(self.l(t)-norm_x)*x - self.damping(t)*v

    def accel(self,x,v,t):
        total_F = self.g(t) + self.spring_f(x,v,t)
        return total_F/self.mass(t)
    
    def rk4_step(self,t, dt): 
        x1 = self.pos()
        v1 = self.vel()
        a1 = self.accel(x1, v1, t)
    
        x2 = self.pos() + 0.5*v1*dt
        v2 = self.vel() + 0.5*a1*dt
        a2 = self.accel(x2, v2, t+dt/2.0)
    
        x3 = self.pos() + 0.5*v2*dt
        v3 = self.vel() + 0.5*a2*dt
        a3 = self.accel(x3, v3, t+dt/2.0)
    
        x4 = self.pos() + v3*dt
        v4 = self.vel() + a3*dt
        a4 = self.accel(x4, v4, t+dt)
    
        xf = self.pos() + (dt/6.0)*(v1 + 2*v2 + 2*v3 + v4)
        vf = self.vel() + (dt/6.0)*(a1 + 2*a2 + 2*a3 + a4)

        self.pos_is(xf)
        self.vel_is(vf)

        return self.params_at(t+dt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--vel", action="store_true",default=True)
    parser.add_argument("--mass", action="store_true",default=False)
    parser.add_argument("--l", action="store_true",default=False)
    parser.add_argument("--g", action="store_true",default=False)
    parser.add_argument("--damping", action="store_true",default=False)
    parser.add_argument("--k", action="store_true",default=False)
    args = parser.parse_args()

    n_features = 2 
    if(args.vel):
        n_features = n_features+2
    if(args.mass):
        n_features = n_features+1
    if(args.l):
        n_features = n_features+1
    if(args.g):
        n_features = n_features+2
    if(args.damping):
        n_features = n_features+1
    if(args.k):
        n_features = n_features+1

    # SETUP
    dt = 1.0/30.0;
    T = 240  
    N = 1000 

    sim_data = np.zeros([N,T,n_features])
    simulator = ParticleSimulator(
                    output_vel=args.vel,
                    output_mass=args.mass,
                    output_l=args.l,
                    output_g=args.g,
                    output_damping=args.damping,
                    output_k=args.k) 
    # initialize the particle around the center
    def pos_init_func():
        x_sample = np.random.uniform(-4.0,4.0)
        y_sample = np.random.uniform(-4.0,4.0)
        while(y_sample**2.0 + x_sample**2.0 > 16.0):
            x_sample = np.random.uniform(-4.0,4.0)
            y_sample = np.random.uniform(-4.0,4.0)
        return [x_sample,y_sample]

    simulator.pos_initializer_is(pos_init_func)
    for s in range(N):
        simulator.reset()
        sim_data[s,0] = simulator.params_at(0)
        for t in range(1,T):
            state = simulator.rk4_step(t,dt)
            sim_data[s,t] = state 

    data_flattened = sim_data.reshape(N,T*n_features)
    torch.save(data_flattened, open('particle_spring_1000_%d.pt'%n_features, 'wb'))

