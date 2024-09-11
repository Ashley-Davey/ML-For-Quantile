# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 18:39:12 2024

@author: ashle
"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt, rcParams
from common import Config, log, function, b_mult, simulation_points, dual_sim, dual_X
from simulate_dual import DualSolver
from simulate_primal import PrimalSolver
from pinn import PINNSolver
import time
from scipy.optimize import minimize
from numpy import array
import pandas as pd
from scipy.interpolate import interp1d
from algo import run
    


rcParams['figure.dpi'] = 600

def value_derivative(data, solver):
    new_data = tf.convert_to_tensor(data, dtype=tf.float64)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        t_data   = new_data[:, :1]
        lam_data = new_data[:,1:2]
        state_data   = new_data[:,2:3]
        mu_data  = new_data[:,3: ]
        tape.watch(state_data )
        tracked_data = tf.concat([t_data, lam_data, state_data, mu_data], 1)  
        v = solver.value_func(tracked_data) 
    return tape.gradient(v, state_data)


    


def main(**kwargs):
        
    #plot example utilities
    
    kwargs_arr = [{'lam': l} for l in [0.2, 2.0]]        
    for kwargs in kwargs_arr:
        Config(**kwargs).plot()
    
        
        
###############################################################################

    #plot distribution of #mu_T
    sigmas = [0.2, 0.02]
    styles  = ['b-', 'g--']
    
    if len(sigmas) > 0:
        fig, ax = plt.subplots(1)

    
        for sigma, style in zip(sigmas, styles):
            config = Config(**{'sigma': np.array([[sigma]])})
            
            paths = np.random.normal(size=[config.simulation_size, config.d, config.primal_time_steps]) 
            dw = paths * config.sqrt_delta_t
            ones_mat = tf.ones(shape=tf.stack([tf.shape(dw)[0], 1, 1]), dtype=tf.float64)
            ones_d = tf.ones(shape=tf.stack([tf.shape(dw)[0], config.d]), dtype=tf.float64)
            ones_vec = tf.ones(shape=tf.stack([tf.shape(dw)[0], 1]), dtype=tf.float64)
            zeros_vec = tf.ones(shape=tf.stack([tf.shape(dw)[0], 1]), dtype=tf.float64)
            
            mu = ones_d * config.mu0 # M x d
            for i in range(config.primal_time_steps):
                ps1 = tf.where(mu <= config.mu_h * ones_vec,
                                    tf.where(mu >= config.mu_l * ones_vec,
                                              b_mult(config.sigma_inv * ones_mat,  (mu - config.mu_l) * (config.mu_h - mu)),
                                              zeros_vec
                                              ),
                                    zeros_vec
                                    )[:, :, np.newaxis]
                
                mu = mu + b_mult(ps1, dw[:,:,i])
                
                mu = tf.minimum(tf.maximum(mu, config.mu_l), config.mu_h)
                
            mu = mu[:, 0]
            
                
            pd.DataFrame({sigma: mu}).plot(kind='density', ax = ax, style = style)
            
            
        ax.set_title("$\hat{\mu}_T$")
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel('Density')
        ax.grid(0.5)
        ax.set_xlim(0.0, 0.15)
        ax.legend()
        plt.show()
    
###############################################################################    

    #run algorithms once
    config = Config(**kwargs)
   
    epsilons = np.linspace(0.0, 0.5, 10)
    for i, epsilon in enumerate(epsilons):
        lambda_star, X_star, y_star, n, F = run(epsilon, **kwargs)
                      
               
    tf.keras.backend.clear_session()
    tf.keras.backend.set_floatx('float64')
    solver = DualSolver(config)
    data, train_data = solver.train()
       
   
    tf.keras.backend.clear_session()
    tf.keras.backend.set_floatx('float64')
    solver = PrimalSolver(config)
    data, train_data = solver.train()
   

    tf.keras.backend.clear_session()
    tf.keras.backend.set_floatx('float64')
    solver_d = PINNSolver(config)
    data, train_data = solver_d.train()
   
       
                        
        
if __name__ == '__main__':
    main(**{})
    log('done')
    
    
    