# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 19:07:12 2024

@author: ashle
"""

from common import Config, log
import numpy as np
from scipy.optimize import minimize
import tensorflow as tf
from matplotlib import pyplot as plt, rcParams

rcParams['figure.dpi'] = 600

def dist(A):
    plt.figure()
    plt.hist(A, bins = 100, density = True)
    plt.grid(0.5)
    plt.show()
    return



def run(epsilon, **kwargs):
    config = Config(**kwargs)        
    
    c_z = config.u_tilde_0(tf.cast(0.0, tf.float64))
    c_z_tilde = config.u_tilde
    

    normals = np.random.normal(size=config.simulation_size) * np.sqrt(config.T)
        
    p = (config.mu0[0] - config.mu_l) / (config.mu_h - config.mu_l)
    psi = p / (1 - p)
    
    
    theta = (config.mu_h - config.mu_l) / config.sigma[0,0]
    theta_l = (config.mu_l - config.r) / config.sigma[0,0]
    
    
    Phi = psi * np.exp(theta * normals - 0.5 * (theta ** 2) * (config.T))
    
    
    F = (1 + Phi) / (1 + psi)
    
    H = np.exp(- theta_l * normals - (config.r + 0.5 * theta_l ** 2) * (config.T)) / (1 + Phi)
    
    Y = (1 + psi) * H
    
    def f(h):
        exp = np.mean((H > h) * F)
        return np.square(exp - epsilon)
    
    H_star = minimize(f, [np.quantile(H, 1 - epsilon)], method='Nelder-Mead').x[0]

    
    threshold       = c_z       / (H_star * (1 + psi))
    threshold_tilde = c_z_tilde / (H_star * (1 + psi))
    
    
    
    def X_3(y):
        return np.where(
                    H < c_z_tilde / (y * (1 + psi)),
                    config.theta + config.I_1(y * (1 + psi) * H),
                    np.where(
                        H < H_star,
                        config.L * np.ones_like(H),
                        np.zeros_like(H)
                        )
                    )
    
    def f(y):
        exp = np.mean(X_3(y) * Y * F)
        return np.square(exp - config.x0)
    
    
    y_3 = minimize(f, [max(2 * (1 - epsilon),1)], bounds = [(0.0, np.inf)], method='Nelder-Mead').x[0]
    
    if y_3 > threshold_tilde:
        lambda_star = (y_3 * (1 + psi) * H_star * config.L - config.U(config.L) - config.U_2(config.theta)).numpy()
        X_star = X_3(y_3)
        n = 3
        y_star = y_3
        
    
    def X_2(y):
        return np.where(
                    H < H_star,
                    config.theta + config.I_1(y * (1 + psi) * H),
                    np.zeros_like(H)
                    )
    
    def f_2(y):
        exp = np.mean(X_2(y) * Y * F)
        return np.square(exp - config.x0)
        
        
    y_2 = minimize(f_2, [max(2 * (1 - epsilon),1)], bounds = [(0.0, np.inf)], method='Nelder-Mead').x[0]
    
    if y_2 > threshold and y_3 <= threshold_tilde:
        
        def f(z):
            return np.square(config.dU_1(z - config.theta) - y_2 * (1 + psi) * H_star)
        
        z_tilde_0 = minimize(f, [config.theta * 1.1], bounds = [(config.theta, np.inf)], method='Nelder-Mead').x[0]
        c_z_tilde_0 = y_2 * (1 + psi) * H_star
        
        
        lambda_star = (z_tilde_0 * c_z_tilde_0 - config.U_1(z_tilde_0 - config.theta) - config.U_2(config.theta)).numpy()
        X_star = X_2(y_2)
        n = 2
        y_star = y_2
    
    def X_1(y):
        return np.where(
                    H < c_z / (y * (1 + psi)),
                    config.theta + config.I_1(y * (1 + psi) * H),
                    np.zeros_like(H)
                    )
    
    def f_1(y):
        exp = np.mean(X_1(y) * Y * F)
        return np.square(exp - config.x0)
        
        
    y_1 = minimize(f_1, [max(2 * (1 - epsilon),1)], bounds = [(0.0, np.inf)], method='Nelder-Mead').x[0]
    
    if y_2 <= threshold and y_3 <= threshold_tilde:
        lambda_star = tf.cast(0.0, tf.float64)
        X_star = X_1(y_1) 
        y_star = y_1
        n = 1

    return lambda_star, X_star, y_star, n, F

def main():
    for epsilon in np.linspace(0.0, 1.0, 5):
        lambda_star, X_star, y_star, n, F = run(epsilon)

        
        
        

if __name__ == '__main__':
    
    main()
    log('done')
    
