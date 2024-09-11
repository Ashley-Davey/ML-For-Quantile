# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 17:29:14 2024

@author: ashle
"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt, rcParams
from scipy.optimize import minimize
from scipy.stats import norm
import time
from scipy.integrate import quad


rcParams['figure.dpi'] = 600

#some general utility functions

def log(*args, **kwargs): #pretty print
    now = time.strftime("%H:%M:%S")
    print("[" + now + "] ", end="")
    print(*args, **kwargs)

def b_mult(A, B): #batch matrix * vector multiplication
    assert len(A.shape) - 1 == len(B.shape) == 2
    return tf.squeeze(A @ tf.expand_dims(B, -1), -1) # @ = matmul

def b_dot(A,B): #batch vector dot product
    assert len(A.shape) == len(B.shape) == 2
    return tf.reduce_sum(A * B, 1, keepdims = True) # * = elementwise mult

def penalise_range(x, lower = -np.inf, upper = np.inf): #loss function for keeping something in a range
    return  10 * tf.square(tf.where(
        tf.logical_or(x > upper * tf.ones_like(x), x < lower * tf.ones_like(x)),
        tf.maximum(x - upper, lower - x),
        0 
        ))




class Config(object):
    def __init__(self, dist = 'discrete', **kwargs):
        #default parameters
        self.T = 1.0 
        self.theta = 1.5 
        self.L = 0.9
        self.x0 = 1.0
        self.r = 0.05 
        
        self.dist = dist
        
        
        if self.dist == 'normal':
            # log('Configured for normal distribution')
            self.mu0 = np.array([0.04, 0.08])
            self.Sigma0 = np.array([[0.002, 0.005], [0.001, 0.001]])
            self.sigma = np.array([[0.2, 0.15], [-0.15, 0.2]])
        elif self.dist == 'discrete': 
            # log('Configured for discrete distribution')
            self.mu0 = np.array([0.07])
            self.sigma = np.array([[0.2]])
            self.mu_l = 0.03
            self.mu_h = 0.1

        
        
        self.lam = 0.2
        self.p1 = 0.5  # U_1 = x ** p1
        self.p2 = 0.3
        self.K = 1.0 #U_2 = K * x ** p2
        
        self.y_max  = 2.5
        self.y_min  = 0.8
        self.lam_min = 0.0
        self.lam_max = 2.5
        self.lam_step = 51 #51
        self.x_min = 0.2
        self.x_max = 2.0
        
        self.weight = 1.0


        self.mollifier = 0.0
        self.function_size   = 100
        self.function_points = 5
        self.bsde_batch_size = 1000 #M
        self.bsde_time_steps = 100 #N
        
        
        self.learning_rate = 0.1
        self.pinn_rate = 0.01
        self.pinn_rate_primal = 0.01
        self.pinn_rate_control = 0.01
        self.iteration_steps = 200
        self.pinn_steps = 10000 #50000
        self.pinn_steps_control = 50000
        
        self.pinn_steps_value = 100000
        self.pinn_steps_constraint = 100000
        self.simulation_size = 100000 #M
        self.hybrid_size = 10000 #M
        self.dual_batch_size = 50000 #M
        self.time_steps = 100 #N        
        self.primal_time_steps = 20 #N        
        self.dual_time_steps = 100 #N      
        
        
        
        self.primal_steps = 500
        self.batch_size = 1000 #M
        self.primal_points = 100
        
        
        self.PINN_value_coll_size  = 2000
        self.PINN_value_bound_size = 200
        self.PINN_const_coll_size  = 1000
        self.PINN_const_bound_size = 100

        self.primal_PINN_value_coll_size  = 2000
        self.primal_PINN_value_bound_size = 200
        self.primal_PINN_const_coll_size  = 5000
        self.primal_PINN_const_bound_size = 500

        self.control_PINN_control_coll_size  = 2000
        self.control_PINN_value_coll_size    = 2000
        self.control_PINN_value_bound_size   = 200
        self.control_PINN_const_coll_size    = 2000
        self.control_PINN_const_bound_size   = 200


        self.hidden = 20 #dual pinn: 100
        self.primal_hidden = 10 #dual pinn: 100
        
        self.bins = 300
        
        
        
        
        
        
        
        #change parameters with constructor 
        for key, value in kwargs.items():
            if hasattr(self, key) and key != 'dist': 
                log(f'Changed {key} to {value}')
                setattr(self, key, value)
                
        
        self.y_range = [self.y_min, self.y_max]
        self.x_range = [self.x_min, self.x_max]
        self.lam_range =  [self.lam_min, self.lam_max]
        self.lams = np.linspace(self.lam_min, self.lam_max, self.lam_step)
        
            
        self.delta_t = self.T / self.time_steps
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.dual_delta_t = self.T / self.dual_time_steps
        self.dual_sqrt_delta_t = np.sqrt(self.dual_delta_t)
        self.display_step = max(int(self.iteration_steps / 2), 1)
        self.display_primal = max(int(self.primal_steps / 2), 1)
        self.display_pinn = max(int(self.pinn_steps / 2), 1)                
        self.display_pinn_control = max(int(self.pinn_steps_control / 2), 1)                
            
        self.lam_tf = tf.Variable(self.lam, dtype = tf.float64, trainable = False)
        
        assert self.theta >= self.L
        self.d = len(self.mu0)
        if self.dist == 'normal': 
            assert self.d == len(self.Sigma0[0])
        assert self.d == len(self.sigma[0])
        self.sigma_inv  = np.linalg.inv(self.sigma)
        self.sigma_sigmaT = self.sigma @ self.sigma.T
        self.sigma_sigmaT_inv = np.linalg.inv(self.sigma_sigmaT)
        self.risk0 = self.sigma_inv @ (self.mu0 - self.r)
        
        ### normal dist
        if self.dist == 'normal':
            self.Sigma0_inv = np.linalg.inv(self.Sigma0)
            self.mu_range = [(mu * 0.9, mu * 1.1) for mu in self.mu0]
        elif self.dist == 'discrete':
            self.mu_range = [(self.mu_l, self.mu_h)]
        
        #discrete dist
        
        

        # self.lam_sample = np.linspace(self.lam_range[0], self.lam_range[1], self.lam_step)
        # self.x_sample = np.ones_like(self.lam_sample ) * self.x0 #np.linspace(self.x_range[0], self.x_range[1], self.lam_step)
        # # self.mu_sample = [np.linspace(mu_range[0], mu_range[1], self.lam_step) for mu_range in self.mu_range]
        # self.mu_sample = [ np.ones_like(self.lam_sample ) * self.mu0[0] for mu_range in self.mu_range]

        assert self.p1 == 0.5

        if self.theta > 0:
            self.z_tilde = self.theta + (np.sqrt((self.U_2(self.theta - self.L)) ** 2 + (self.theta - self.L)) - self.U_2(self.theta - self.L)) ** 2
        else:
            self.z_tilde = 0.0
            
              
        self.u_tilde = self.dU_1(self.z_tilde - self.theta + 1e-8)
        

    def dual_value(self, t, y, mu, lam):
        p = (mu - self.mu_l) / (self.mu_h - self.mu_l)
        if np.isclose(p, 0.0) or np.isclose(p, 1.0):
            
            theta = (mu - self.r) / self.sigma[0,0]
            
            def f(x):
                Y = y * np.exp(- theta * x - (self.r + 0.5 * theta ** 2) * (self.T - t))
                return self.V(Y, lam) 
        else:
            psi = p / (1 - p)
            theta = (self.mu_h - self.mu_l) / self.sigma[0,0]
            theta_l = (self.mu_l - self.r) / self.sigma[0,0]
            
            def f(x):
                F = (1 + psi * np.exp(theta * x - 0.5 * (theta ** 2) * (self.T - t))) / (1 + psi)
                Y = y * np.exp(- theta_l * x - (self.r + 0.5 * theta_l ** 2) * (self.T - t))
                return F * self.V(Y / F, lam) 
            
        rands = np.random.normal(size = self.simulation_size) * np.sqrt(self.T - t)
        return np.mean(f(rands))
        

    def dual_derivative(self, t, y, mu, lam):
        p = (mu - self.mu_l) / (self.mu_h - self.mu_l)
        if np.isclose(p, 0.0) or np.isclose(p, 1.0):
            
            theta = (mu - self.r) / self.sigma[0,0]
            
            def f(x):
                Y = np.exp(- theta * x - (self.r + 0.5 * theta ** 2) * (self.T - t))
                return Y * self.V_y(y * Y, lam)
        else:
            psi = p / (1 - p)
            theta = (self.mu_h - self.mu_l) / self.sigma[0,0]
            theta_l = (self.mu_l - self.r) / self.sigma[0,0]
            
            def f(x):
                F = (1 + psi * np.exp(theta * x - 0.5 * (theta ** 2) * (self.T - t))) / (1 + psi)
                Y = np.exp(- theta_l * x - (self.r + 0.5 * theta_l ** 2) * (self.T - t))
                return Y * self.V_y(y * Y / F, lam)
        
        rands = np.random.normal(size = self.simulation_size) * np.sqrt(self.T - t)
        return np.mean(f(rands))


    
    def dual_constraint(self, t, y, mu, lam):
        p = (mu - self.mu_l) / (self.mu_h - self.mu_l)
        if np.isclose(p, 0.0) or np.isclose(p, 1.0):
            
            theta = (mu - self.r) / self.sigma[0,0]
            
            def f(x):
                Y = y * np.exp(- theta * x - (self.r + 0.5 * theta ** 2) * (self.T - t))
                return self.constraint(Y, lam)
        else:
            psi = p / (1 - p)
            theta = (self.mu_h - self.mu_l) / self.sigma[0,0]
            theta_l = (self.mu_l - self.r) / self.sigma[0,0]
            
            def f(x):
                F = (1 + psi * np.exp(theta * x - 0.5 * (theta ** 2) * (self.T - t))) / (1 + psi)
                Y = y * np.exp(- theta_l * x - (self.r + 0.5 * theta_l ** 2) * (self.T - t))
                return F * self.constraint(Y / F, lam)
        
        rands = np.random.normal(size = self.simulation_size) * np.sqrt(self.T - t)
        return np.mean(f(rands))
        
        
        
        
    def k(self, lam):
        return (self.U(self.L) + lam + self.U_2(self.theta)) / (self.L + 1e-8 )
    
    
    def z_tilde_0(self, lam):
        if self.theta == 0:
            return tf.zeros_like(lam)
        else:
            return tf.where(self.k(lam) > self.u_tilde * tf.ones_like(lam),
                            tf.zeros_like(lam),
                            self.theta + (tf.sqrt((self.U_2(self.theta) + lam) ** 2 + self.theta) - (self.U_2(self.theta) + lam)) ** 2
                            )
            
    def u_tilde_0(self, lam):
        return self.dU_1(self.z_tilde_0(lam) - self.theta + 1e-8)
        
        
    def plot(self):
        xaxis = np.linspace(0, 5.0, 10000)
        yaxis = np.linspace(0.1, 5.0, 10000)
        lamaxis = np.ones_like(xaxis) * self.lam

        #primal plot
        plt.figure()
        plt.grid(0.5)
        plt.xlabel('$x$')
        plt.ylabel('$U$')
        plt.plot(xaxis, self.U_lam(xaxis, lamaxis), label = '$U_\lambda$')
        plt.plot(xaxis, self.U_conc(xaxis, lamaxis), linestyle = 'dashed', label = '$U_\lambda^c$')
        plt.axvline(self.theta, label = '$\\theta$', linestyle = 'dotted', c = 'C3')
        plt.axvline(self.L, label = '$L$', linestyle = 'dotted', c = 'C4')
        plt.legend()
        
        plt.figure()
        plt.grid(0.5)
        plt.xlabel('$x$')
        plt.ylabel("$U'$")
        plt.plot(xaxis, self.U_conc_x(xaxis, lamaxis), label = "$(U_\lambda^c)'$")
        plt.axvline(self.theta, label = '$\\theta$', linestyle = 'dotted', c = 'C3')
        plt.axvline(self.L, label = '$L$', linestyle = 'dotted', c = 'C4')
        plt.legend()
        
        plt.figure()
        plt.grid(0.5)
        plt.xlabel('$x$')
        plt.ylabel("$h$")
        plt.plot(xaxis, self.constraint_primal(xaxis), label = "constraint")
        plt.axvline(self.L, label = '$L$', linestyle = 'dotted', c = 'C4')
        plt.legend()       
        
        #dual plot
        plt.figure()
        plt.grid(0.5)
        plt.xlabel('$y$')
        plt.ylabel('$V$')
        plt.plot(yaxis, self.V(yaxis, lamaxis), label = '$V_\lambda^c$')
        plt.plot(yaxis, self.V_exp(yaxis, lamaxis), label = '$V_\lambda^c$ (explicit)', linestyle = 'dashed')
        
        
        if self.k(self.lam_tf) > self.u_tilde:
            plt.axvline(self.k(self.lam_tf), label = '$k_\lambda$', linestyle = 'dotted', c = 'C2')
            plt.axvline(self.u_tilde, label = "$U'(\\tilde{z})$", linestyle = 'dotted', c = 'C3')
        else: 
            plt.axvline(self.u_tilde_0(self.lam_tf), label = "$U'(\\tilde{z}_0)$", linestyle = 'dotted', c = 'C2')
        plt.xlim(-0.1, yaxis[-1]*1.02)
        plt.legend()
        
        plt.figure()
        plt.grid(0.5)
        plt.xlabel('$y$')
        plt.ylabel("$-V'$")
        plt.plot(yaxis, -self.V_y(yaxis, lamaxis), label = "$-(V_\lambda^c)'$")
        if self.k(self.lam_tf) > self.u_tilde:
            plt.axvline(self.k(self.lam_tf), label = '$k_\lambda$', linestyle = 'dotted', c = 'C2')
            plt.axvline(self.u_tilde, label = "$U'(\\tilde{z})$", linestyle = 'dotted', c = 'C3')
            plt.axhline(self.z_tilde, label = "$\\tilde{z}$", linestyle = 'dotted', c = 'C3')
        else: 
            plt.axvline(self.u_tilde_0(self.lam_tf), label = "$U'(\\tilde{z}_0)$", linestyle = 'dotted', c = 'C2')
            plt.axhline(self.z_tilde_0(self.lam_tf), label = "$\\tilde{z}_0$", linestyle = 'dotted', c = 'C2')
        plt.xlim(-0.1, yaxis[-1]*1.02)
        plt.legend()
        
        plt.figure()
        plt.grid(0.5)
        plt.xlabel('$y$')
        plt.ylabel("h")
        plt.plot(yaxis, self.constraint(yaxis, lamaxis), label = 'constraint')
        if self.k(self.lam_tf) > self.u_tilde:
            plt.axvline(self.k(self.lam_tf), label = '$k_\lambda$', linestyle = 'dotted', c = 'C2')
            plt.axvline(self.u_tilde, label = "$U'(\\tilde{z})$", linestyle = 'dotted', c = 'C3')
        else: 
            plt.axvline(self.u_tilde_0(self.lam_tf), label = "$U'(\\tilde{z}_0)$", linestyle = 'dotted', c = 'C2')
        plt.xlim(-0.1, yaxis[-1]*1.02)
        plt.legend()        
        plt.show()
                
    # @tf.autograph.experimental.do_not_convert            
    def psi(self, t, mu): # M x d x d
        if self.dist == 'normal': 
            ones_mat = tf.ones(shape=tf.stack([tf.shape(mu)[0], 1, 1]), dtype=tf.float64)
            psi = tf.linalg.inv(
                (self.Sigma0_inv * ones_mat  + self.sigma_sigmaT_inv * tf.expand_dims(t, -1) * ones_mat)
                ) @ (self.sigma_inv.T * ones_mat)
            return psi
        elif self.dist == 'discrete':
            #mu: M x 1 -> M x 1 x 1
            
            ones_mat = tf.ones(shape=tf.stack([tf.shape(mu)[0], 1, 1]), dtype=tf.float64)
            ones_vec = tf.ones(shape=tf.stack([tf.shape(mu)[0], 1]), dtype=tf.float64)
            zeros_vec = tf.zeros(shape=tf.stack([tf.shape(mu)[0], 1]), dtype=tf.float64)


            return tf.where(mu <= self.mu_h * ones_vec,
                            tf.where(mu >= self.mu_l * ones_vec,
                                     b_mult(self.sigma_inv * ones_mat,  (mu - self.mu_l) * (self.mu_h - mu)),
                                     zeros_vec
                                     ),
                            zeros_vec
                            )[:, :, np.newaxis]
        

    #tensorflow functions
    
    def U_1(self, z):
        z = tf.cast(z, tf.float64)
        return tf.pow(tf.maximum(z, 1e-8), self.p1)
    
    def dU_1(self, z):
        z = tf.cast(z, tf.float64)
        return self.p1 * tf.pow(tf.maximum(z, 1e-8), self.p1 - 1)
    
    def I_1(self, y):
        return  tf.pow(y / self.p1, 1 / (self.p1 - 1))
    
    def I_1_y(self, y):
        return (1 / (self.p1 - 1)) *  tf.pow(y / self.p1, (1 / (self.p1 - 1)) - 1)
    
    def U_2(self, z):
        z = tf.cast(z, tf.float64)
        return self.K * tf.pow(tf.maximum(z, 1e-8), self.p2)
                    
    def U(self, x):
        x = tf.cast(x, tf.float64)
        return tf.where(x >= self.theta * tf.ones_like(x),
                        self.U_1(x - self.theta),
                        - self.U_2(self.theta - x)
                        )
    
    def U_lam(self, x, lam):
        return self.U(x) + tf.where(x >= self.L * tf.ones_like(x), lam, tf.zeros_like(x))
    
    def U_conc(self, x, lam):
        return tf.where(self.k(lam) > self.u_tilde * tf.ones_like(lam),
                        tf.where(x < tf.zeros_like(x),
                            10.0 * x,
                            tf.where(x < self.L * tf.ones_like(x),
                                     self.k(lam) * x - self.U_2(self.theta),
                                     tf.where(x < self.z_tilde * tf.ones_like(x),
                                              self.u_tilde * (x - self.L) - self.U_2(self.theta - self.L) + lam,
                                              self.U_1(x - self.theta) + lam
                                              )
                                     )   
                            ),
                        tf.where(x < 0 * tf.ones_like(x),
                            10.0 * x,
                            tf.where(x < self.z_tilde_0(lam),
                                     self.u_tilde_0(lam) * x - self.U_2(self.theta),
                                     self.U_1(x - self.theta) + lam
                                     )
                            )
                        )
                        
        
    def U_conc_x(self, x, lam):
        return tf.where(self.k(lam) > self.u_tilde * tf.ones_like(lam),
                        tf.where(x < 0 * tf.ones_like(x),
                            tf.zeros_like(x),
                            tf.where(x < self.L * tf.ones_like(x),
                                     self.k(lam),
                                     tf.where(x < self.z_tilde * tf.ones_like(x),
                                              self.u_tilde * tf.ones_like(x),
                                              self.dU_1(x - self.theta)
                                              )
                                     )   
                            ),
                        tf.where(x < 0 * tf.ones_like(x),
                            tf.zeros_like(x),
                            tf.where(x < self.z_tilde_0(lam),
                                     self.u_tilde_0(lam),
                                     self.dU_1(x - self.theta)
                                     )
                            )
                        )
        
    def x(self, y, lam):
        return tf.where( self.k(lam) > self.u_tilde * tf.ones_like(lam),
                        tf.where(y < self.u_tilde * tf.ones_like(y),
                            self.theta + self.I_1(y),
                            tf.where(y < self.k(lam),
                                     tf.ones_like(y) * self.L,
                                     tf.zeros_like(y)
                                     )
                            ),
                        tf.where(y < self.u_tilde_0(lam),
                            self.theta + self.I_1(y),
                            tf.zeros_like(y)
                            )
                        )
        
    # def constraint2(self, y, lam):
    #     return tf.where( self.k(lam) > self.u_tilde * tf.ones_like(lam),
    #                     tf.sigmoid(100 * ( self.k(lam) - y)),
    #                     tf.sigmoid(100 * ( self.u_tilde_0(lam) - y)) 
    #                     )
    
    def constraint(self, y, lam):
        if np.isclose(self.mollifier, 0.0):
            return tf.where( self.k(lam) > self.u_tilde * tf.ones_like(lam),
                            tf.where(y <= self.k(lam),
                                            tf.ones_like(y),
                                            tf.zeros_like(y)
                                            ),
                            tf.where(y <= self.u_tilde_0(lam),
                                            tf.ones_like(y),
                                            tf.zeros_like(y)
                                            )
                            )
        else:
            return tf.where( self.k(lam) > self.u_tilde * tf.ones_like(lam),
                            tf.sigmoid( ( self.k(lam) - y) / self.mollifier),
                            tf.sigmoid(( self.u_tilde_0(lam) - y) / self.mollifier) 
                            )

    def constraint_primal(self, x):
        if np.isclose(self.mollifier, 0.0):
            return tf.where(x >= tf.ones_like(x) * self.L,
                                            tf.ones_like(x),
                                            tf.zeros_like(x)
                                            )
        else:
            return tf.sigmoid( ( x - self.L) / self.mollifier)
        
    def V(self, y, lam):
        return tf.where(y >= tf.zeros_like(y), 
                            self.U_conc(self.x(y, lam), lam) - self.x(y, lam) * y,
                            y * -100
                            )
    
    def V_exp(self, y, lam):
        return tf.where( self.k(lam) > self.u_tilde * tf.ones_like(lam),
                        tf.where(y < self.u_tilde * tf.ones_like(y),
                            self.U_1(self.I_1(y)) + lam - (self.theta + self.I_1(y)) * y,
                            tf.where(y < self.k(lam),
                                     - self.U_2(self.theta - self.L) + lam - self.L * y,
                                     - self.U_2(self.theta) * tf.ones_like(y)
                                     )
                            ),
                        tf.where(y < self.u_tilde_0(lam),
                            self.U_1(self.I_1(y)) + lam - (self.theta + self.I_1(y)) * y,
                            - self.U_2(self.theta) * tf.ones_like(y)
                            )
                        )
    
    
    
    
    
        # return tf.where(y >= tf.zeros_like(y), 
        #                 tf.where( self.k(lam) > self.u_tilde * tf.ones_like(lam),
        #                         self.U_conc(self.x(y, lam), lam) - self.x(y, lam) * y,
        #                         tf.where(y < self.u_tilde_0(lam),
        #                             self.U_1(self.I_1(y)) + lam - y * (self.theta + self.I_1(y)),
        #                             - self.U_2(self.theta) * tf.ones_like(y)
        #                             )
        #                         ),
        #                 y * -100
        #                 )
                     
                        
    
    def V_y(self, y, lam):
        return -self.x(y, lam)
    
    def V_yy(self, y, lam):
        return tf.where( self.k(lam) > self.u_tilde * tf.ones_like(lam),
                        tf.where(y < self.u_tilde * tf.ones_like(y),
                            - self.I_1_y(y),
                            tf.zeros_like(y)
                            ),
                        tf.where(y < self.u_tilde_0(lam),
                            - self.I_1_y(y),
                            tf.zeros_like(y)
                            )
                        )    
    
    def concavify(self, X, lam):
        return tf.where( self.k(lam) > self.u_tilde * tf.ones_like(lam), #0, L or > ztilde
                        tf.where(X >= self.z_tilde * tf.ones_like(X),
                            X,
                            tf.where(X >= (self.z_tilde + self.L) * tf.ones_like(X) / 2,
                                     tf.ones_like(X) * self.z_tilde,
                                     tf.where(X >= self.L * tf.ones_like(X) / 2,
                                              tf.ones_like(X) * self.L,
                                              tf.zeros_like(X)
                                              )
                                     )
                            ),
                        tf.where(X >= self.z_tilde_0(lam),  #0 or z_tilde_0
                            X,
                            tf.where(X >= self.z_tilde_0(lam) / 2,
                                     self.z_tilde_0(lam),
                                     tf.zeros_like(X)
                                     )
                            )
                        )

    



def function(**kwargs):
    config = Config(**kwargs)
    
    M = config.simulation_size
    d = 100
    N = 100
    
    delta_t = config.T / N
    sqrt_delta_t = np.sqrt(delta_t)
    
    yaxis = np.linspace(config.y_range[0], config.y_range[1], d)[np.newaxis, :]
    
    paths = np.random.normal(size=[M, config.d, N]) 
    dw_sample = paths * sqrt_delta_t
    dw = dw_sample
    
    
    sigma_inv = config.sigma_inv * np.ones((M, config.d, config.d))
    times = np.linspace(0, config.T, N + 1)
    Y = np.ones((M, 1)) 
    
    mu = config.mu0 * np.ones((M, config.d)) 
    for i in range(N):
        
        Y = Y * (
            1 
            - config.r * delta_t
            - np.sum(np.squeeze(sigma_inv @ (mu - config.r)[:, :, np.newaxis], -1) * dw[:,:,i], axis = 1)
            )[:, np.newaxis]
        t = np.ones((M, 1)) * times[i]
        mu = mu + np.squeeze(config.psi(t, mu) @  dw[:,:,i][:, :, np.newaxis], -1)
        if config.dist == 'discrete':
            mu = np.minimum(np.maximum(mu, config.mu_l), config.mu_h)
    
    value = np.mean(config.V(Y @ yaxis, config.lam * np.ones((M, d))), axis = 0) #+ config.x0 * np.squeeze(yaxis, 0)
    v_y = np.mean(Y * config.V_y(Y @ yaxis, config.lam * np.ones((M, d))), axis = 0) #+ config.x0 * np.squeeze(yaxis, 0)
    v_yy = np.mean((Y ** 2 ) * config.V_yy(Y @ yaxis, config.lam * np.ones((M, d))), axis = 0) #+ config.x0 * np.squeeze(yaxis, 0)
    constraint = np.mean(config.constraint(Y @ yaxis, config.lam * np.ones((M, d))), axis = 0)
    
    yaxis = np.squeeze(yaxis, 0)
    
    tf.keras.backend.clear_session()

    return yaxis, value, constraint, v_y, v_yy
    
def function_lam(**kwargs):
    config = Config(**kwargs)
    
    M = 10000
    d = 100
    N = 100
    
    delta_t = config.T / N
    sqrt_delta_t = np.sqrt(delta_t)
    
    yaxis = np.ones((1, d)) * config.y
    lamaxis = np.linspace(config.lam_range[0], config.lam_range[1], d)[np.newaxis, :]
    
    paths = np.random.normal(size=[M, config.d, N]) 
    dw_sample = paths * sqrt_delta_t
    dw = dw_sample
    
    
    sigma_inv = config.sigma_inv * np.ones((M, config.d, config.d))
    times = np.linspace(0, config.T, N + 1)
    Y = np.ones((M, 1)) 
    
    mu = config.mu0 * np.ones((M, config.d)) 
    for i in range(N):
        
        Y = Y * (
            1 
            - config.r * delta_t
            - np.sum(np.squeeze(sigma_inv @ (mu - config.r)[:, :, np.newaxis], -1) * dw[:,:,i], axis = 1)
            )[:, np.newaxis]
        
        t = np.ones((M, 1)) * times[i]
    
        mu = mu + np.squeeze(config.psi(t, mu) @  dw[:,:,i][:, :, np.newaxis], -1)
        if config.dist == 'discrete':
            mu = np.minimum(np.maximum(mu, config.mu_l), config.mu_h)
    value = np.mean(config.V(Y @ yaxis, lamaxis * np.ones((M, d))), axis = 0) #+ config.x0 * np.squeeze(yaxis, 0)
    v_y = np.mean(Y * config.V_y(Y @ yaxis, lamaxis * np.ones((M, d))), axis = 0) #+ config.x0 * np.squeeze(yaxis, 0)
    v_yy = np.mean((Y ** 2 ) * config.V_yy(Y @ yaxis, lamaxis * np.ones((M, d))), axis = 0) #+ config.x0 * np.squeeze(yaxis, 0)
    constraint = np.mean(config.constraint(Y @ yaxis, lamaxis * np.ones((M, d))), axis = 0)
    
    lamaxis = np.squeeze(lamaxis, 0)
    
    tf.keras.backend.clear_session()

    return lamaxis, value, constraint, v_y, v_yy
        
    
def simulation_points(y, lam, **kwargs):
    config = Config(**kwargs)
    
    M = config.simulation_size
    N = 100
    
    delta_t = config.T / N
    sqrt_delta_t = np.sqrt(delta_t)
    
    yaxis = y[np.newaxis, :]
    lamaxis = lam[np.newaxis, :]

    
    paths = np.random.normal(size=[M, config.d, N]) 
    dw_sample = paths * sqrt_delta_t
    dw = dw_sample
    
    sigma_inv = config.sigma_inv * np.ones((M, config.d, config.d))
    times = np.linspace(0, config.T, N + 1)
    Y = np.ones((M, 1)) 
    
    mu = config.mu0 * np.ones((M, config.d)) 
    for i in range(N):
        
        Y = Y * (
            1 
            - config.r * delta_t
            - np.sum(np.squeeze(sigma_inv @ (mu - config.r)[:, :, np.newaxis], -1) * dw[:,:,i], axis = 1)
            )[:, np.newaxis]
        t = np.ones((M, 1)) * times[i]
        mu = mu + np.squeeze(config.psi(t, mu) @  dw[:,:,i][:, :, np.newaxis], -1)
        if config.dist == 'discrete':
            mu = np.minimum(np.maximum(mu, config.mu_l), config.mu_h)
    value = np.mean(config.V(Y @ yaxis, np.ones((M, 1)) @ lamaxis ), axis = 0) #+ config.x0 * np.squeeze(yaxis, 0)
    v_y = np.mean(Y * config.V_y(Y @ yaxis, np.ones((M, 1)) @ lamaxis), axis = 0) #+ config.x0 * np.squeeze(yaxis, 0)

    v_yy = np.mean((Y ** 2 ) * config.V_yy(Y @ yaxis, np.ones((M, 1)) @ lamaxis), axis = 0) #+ config.x0 * np.squeeze(yaxis, 0)
    constraint = np.mean(config.constraint(Y @ yaxis, np.ones((M, 1)) @ lamaxis), axis = 0)
    
    lamaxis = np.squeeze(lamaxis, 0)
    
    tf.keras.backend.clear_session()

    return lamaxis, value, constraint, v_y, v_yy

def dual_sim(y0, lam, mu0, **kwargs):
    
    
    
    config = Config(**kwargs)
    assert config.d == 1
    
    M = config.simulation_size
    N = 100
    n = y0.shape[0]
    
    delta_t = config.T / N
    sqrt_delta_t = np.sqrt(delta_t)
    
    yaxis = y0[np.newaxis, :] #1 x n
    lamaxis = lam[np.newaxis, :] # 1 x n

    
    paths = np.random.normal(size=[M * n, config.d, N]) 
    dw_sample = paths * sqrt_delta_t
    dw = dw_sample
    
    sigma_inv = config.sigma_inv * np.ones((M * n, config.d, config.d))
    times = np.linspace(0, config.T, N + 1)
    Y = np.ones((M * n, 1)) 
    
    mu = (mu0 * np.ones((M, n))).reshape(M * n, 1) 
    for i in range(N):
        
        Y = Y * (
            1 
            - config.r * delta_t
            - np.sum(np.squeeze(sigma_inv @ (mu - config.r)[:, :, np.newaxis], -1) * dw[:,:,i], axis = 1)
            )[:, np.newaxis]
        t = np.ones((M, 1)) * times[i]
        mu = mu + np.squeeze(config.psi(t, mu) @  dw[:,:,i][:, :, np.newaxis], -1)
        if config.dist == 'discrete':
            mu = np.minimum(np.maximum(mu, config.mu_l), config.mu_h)
            
    Y = Y.reshape(M, n)   
    value = np.mean(config.V(Y * yaxis, np.ones((M, 1)) @ lamaxis ), axis = 0) #+ config.x0 * np.squeeze(yaxis, 0)
    v_y = np.mean(Y * config.V_y(Y * yaxis, np.ones((M, 1)) @ lamaxis), axis = 0) #+ config.x0 * np.squeeze(yaxis, 0)

    v_yy = np.mean((Y ** 2 ) * config.V_yy(Y * yaxis, np.ones((M, 1)) @ lamaxis), axis = 0) #+ config.x0 * np.squeeze(yaxis, 0)
    constraint = np.mean(config.constraint(Y * yaxis, np.ones((M, 1)) @ lamaxis), axis = 0)
    
    lamaxis = np.squeeze(lamaxis, 0)
    
    tf.keras.backend.clear_session()

    return lamaxis, value, constraint, v_y, v_yy
        
def dual_X(y0, lam, mu0, **kwargs):
    
    
    
    config = Config(**kwargs)
    assert config.d == 1
    
    M = config.simulation_size
    N = 100
    n = y0.shape[0]
    
    delta_t = config.T / N
    sqrt_delta_t = np.sqrt(delta_t)
    
    yaxis = y0[np.newaxis, :] #1 x n
    lamaxis = lam[np.newaxis, :] # 1 x n

    
    paths = np.random.normal(size=[M * n, config.d, N]) 
    dw_sample = paths * sqrt_delta_t
    dw = dw_sample
    
    sigma_inv = config.sigma_inv * np.ones((M * n, config.d, config.d))
    times = np.linspace(0, config.T, N + 1)
    Y = np.ones((M * n, 1)) 
    
    mu = (mu0 * np.ones((M, n))).reshape(M * n, 1) 

    for i in range(N):
        Y = Y * (
            1 
            - config.r * delta_t
            - np.sum(np.squeeze(sigma_inv @ (mu - config.r)[:, :, np.newaxis], -1) * dw[:,:,i], axis = 1)
            )[:, np.newaxis]
        t = np.ones((M * n, 1)) * times[i]
        mu = mu + np.squeeze(config.psi(t, mu) @  dw[:,:,i][:, :, np.newaxis], -1)
        if config.dist == 'discrete':
            mu = np.minimum(np.maximum(mu, config.mu_l), config.mu_h)
            
        
            
    Y = Y.reshape(M, n)   
    
    return - config.V_y(Y * yaxis, np.ones((M, 1)) @ lamaxis)

    
class FeedForwardSubNet(tf.keras.Model):
    def __init__(self, num_hiddens = [10, 10], dim = 1):
        super(FeedForwardSubNet, self).__init__()
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                    # kernel_regularizer=tf.keras.regularizers.L1(0.01),
                                                    # bias_regularizer=tf.keras.regularizers.L1(0.01),
                                                    use_bias=True,
                                                    activation=None)
                              for i in range(len(num_hiddens))]
        self.dense_layers.append(tf.keras.layers.Dense(dim, 
                                                        # kernel_regularizer=tf.keras.regularizers.L1(0.01),
                                                        # bias_regularizer=tf.keras.regularizers.L1(0.01),
                                                        use_bias=True, 
                                                        activation=None))

    def call(self, x, training):
      
        scaler = ( 
            tf.linalg.diag(
                tf.concat(
                    [ 
                        tf.ones(shape=[3], dtype=tf.float64),
                        (1 / 0.04) * tf.ones(shape=[tf.shape(x)[1] - 3], dtype=tf.float64) 
                    ],
                    axis = 0
                    )
            ) 
            * tf.ones(shape=tf.stack([tf.shape(x)[0], 1, 1]), dtype=tf.float64)
            )
                
        x = b_mult(scaler, x)
            
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x =  tf.tanh(x)
        x = self.dense_layers[-1](x)
        return x 
    

    
class ConstantModel(tf.keras.Model):
    def __init__(self, dim = 1):
        super(ConstantModel, self).__init__()
        self.constant = tf.Variable(tf.random_normal_initializer( mean=0.0, stddev=0.05, seed=None)(shape=[dim], dtype=tf.float64))

    def call(self, x, training):
        return tf.ones(shape=tf.stack([tf.shape(x)[0], 1]), dtype = tf.float64) @ self.constant[np.newaxis, :]
    
    
