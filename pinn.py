# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:01:33 2024

@author: Charlotte
"""

import tensorflow as tf
import numpy as np
from common import log, b_dot, b_mult, Config, FeedForwardSubNet, penalise_range, function, function_lam, simulation_points
import time
from matplotlib import pyplot as plt, rcParams
import itertools
from scipy.optimize import minimize


rng = np.random.default_rng()
tf .get_logger().setLevel('ERROR') #remove warning messages for taking second derivatives
rcParams['figure.dpi'] = 600


 
#solver

class PINNSolver(object):
    def __init__(self, config):
        log('Running Dual PINN Algorithm')
        self.start_time = time.time()
        self.config = config
        self.value = ValueModel(config)
        # self.constraint = ConstraintModel(config)
        self.optimiser = tf.keras.optimizers.Adam(learning_rate=self.config.pinn_rate) 
        self.value_sample = self.get_sample(self.config.PINN_value_coll_size, self.config.PINN_value_bound_size)
        # self.constraint_sample = self.get_sample(self.config.PINN_const_coll_size, self.config.PINN_const_bound_size)

    def get_sample(self, coll_size, bound_size):
        ret = {}
        
        y_points  = [rng.uniform(self.config.y_range[0], self.config.y_range[1], coll_size)]
        lam_points  = [rng.uniform(self.config.lam_range[0], self.config.lam_range[1], coll_size)]
        mu_points = [rng.uniform(self.config.mu_range[i][0], self.config.mu_range[i][1], coll_size) for i in range(self.config.d)]
        t_points  = [rng.uniform(0, self.config.T, coll_size) ]
        ret['collocation'] = np.stack(t_points + lam_points + y_points + mu_points, -1)
        
        y_points  = [rng.uniform(self.config.y_range[0], self.config.y_range[1], bound_size)]
        lam_points  = [rng.uniform(self.config.lam_range[0], self.config.lam_range[1], bound_size)]
        mu_points = [rng.uniform(self.config.mu_range[i][0], self.config.mu_range[i][1], bound_size) for i in range(self.config.d)]
        t_points  = [np.ones(bound_size) * self.config.T]
        
        ret['boundary'] = np.stack(t_points + lam_points + y_points + mu_points, -1)
                
        return ret
    
    # def val_data(self, num_points, num_size):
    #     '''

    #     num_size points to plot y -> f(y, lam) and lam -> f(y, lam) for num_points different values of y/lam

    #     '''
    #     y_min = self.config.y_range[0]
    #     y_max = self.config.y_range[1]
    #     lam_min = self.config.lam_range[0]
    #     lam_max = self.config.lam_range[1]
        
    #     y_array   = np.linspace(y_min, y_max, num_size)
    #     lam_array = np.linspace(lam_min, lam_max, num_size)
    #     y_points   = np.linspace(y_min, y_max, num_points)
    #     lam_points = np.linspace(lam_min, lam_max, num_points)
        
        
    #     y_points1   = [np.tile(y_array, num_points)]
    #     lam_points1 = [np.repeat(lam_points, num_size)]
    #     y_points2   = [np.repeat(y_points, num_size)]
    #     lam_points2 = [np.tile(lam_array, num_points)]
        
    #     mu_points = [np.ones(num_size * num_points) * mu for mu in self.config.mu0]
    #     t_points  = [np.zeros(num_size * num_points) ]
        
    #     ret  = np.stack(t_points + lam_points1 + y_points1 + mu_points, -1)
    #     ret2 = np.stack(t_points + lam_points2 + y_points2 + mu_points, -1)
    #     #points 1
    #     #0 y1 lam1 mu1 mu2
    #     #0 y2 lam1 mu1 mu2
    #     #...
    #     #0 yN lam1 mu1 mu2
    #     #0 y1 lam2 mu1 mu2
    #     #etc
    #     #points 2 swap y and lam
        
    #     return np.concatenate((ret, ret2), 0)
    
    
    def train(self):
        
        # begin sgd iteration
        
        data = {
            'value loss': [],
            'constraint loss': [],
            'times': [],
            }
        train_data = {}
        
        try:
            try:
                step=0    
                with tf.GradientTape(persistent = True) as tape:
                    value_loss = self.value(self.value_sample, tape = tape) 
                    # constraint_loss = self.constraint(self.constraint_sample, tape = tape) 
                log(f"Step: {0:6d} \t Time: {time.time() - self.start_time:5.2f} ")
                self.optimiser.build(self.value.trainable_variables)# + self.constraint.trainable_variables)
                while (len(data['value loss']) < 1000 or data['value loss'][-1]  > 5e-5) and step < self.config.pinn_steps:
                    step += 1
                    display = step % self.config.display_pinn == 0 or step == 1
                    value_loss = self.train_step()
                    data['value loss'].append(value_loss)
                    # data['constraint loss'].append(constraint_loss)
                    data['times' ].append(time.time() - self.start_time)
                    if display:
                        log(f"Step: {step:6d} \t Time: {time.time() - self.start_time:5.2f} \t  v_loss: {value_loss:+.3e}")
                
            except Exception as e:
                    log('Termination due to ', e)
                    pass
                
        except BaseException:
            log('Terminated Manually')
            pass
    
        log(f"Ended at {step} iterations")
        log(f"Step: {step:6d} \t Time: {time.time() - self.start_time:5.2f} \t  v_loss: {value_loss:+.3e}")
        tf.keras.backend.clear_session()
        return data, train_data


    @tf.function
    def train_step(self):
        with tf.GradientTape(persistent = True) as tape:
            v_loss = self.value(self.value_sample, tape = tape)
            # h_loss = self.constraint(self.constraint_sample, tape = tape)
        # h_grad = tape.gradient(h_loss, self.constraint.trainable_variables)
        v_grad =  tape.gradient(v_loss, self.value.trainable_variables)
        del tape
        self.optimiser.apply_gradients(zip(v_grad, self.value.trainable_variables))
        # self.optimiser.apply_gradients(zip(h_grad, self.constraint.trainable_variables))
        return v_loss
        
    def value_func(self, *args, **kwargs):
        return self.value.net(*args, **kwargs)
    
    # def constraint_func(self, *args, **kwargs):
    #     return self.constraint.net(*args, **kwargs)
    
                        
    
    
class PINNModel(tf.keras.Model):
    def __init__(self, config):
        super(PINNModel, self).__init__()
        self.config = config
        self.net = FeedForwardSubNet([config.hidden, config.hidden], 1)        
        
    def L(self, t, y, mu, f_t, f_y, f_m, f_yy, f_ym, f_mm):
        ones_mat = tf.ones(shape=tf.stack([tf.shape(y)[0], 1, 1]), dtype=tf.float64)
        sigma_inv = self.config.sigma_inv * ones_mat
        risk = b_mult(sigma_inv, mu - self.config.r)
        
        psi = self.config.psi(t, mu)
        
        return (
            f_t
            - self.config.r * y * f_y
            + 0.5 * tf.square(y) * b_dot(risk, risk) * f_yy
            + 0.5 * tf.linalg.trace((psi @ tf.transpose(psi , perm=[0,2,1])) @ f_mm)[:, np.newaxis]
            - y * b_dot(f_ym, b_mult(psi, risk))
            ) 
    
    def collocation_loss(self, data, tape):
        t_data   = data[:, :1]
        lam_data = data[:,1:2]
        y_data   = data[:,2:3]
        mu_data  = data[:,3: ]
                          
        tape.watch(t_data )
        tape.watch(y_data )
        tape.watch(mu_data)
        tape.watch(lam_data)
        
        tracked_data = tf.concat([t_data, lam_data, y_data, mu_data], 1)  
        
        f = self.net(tracked_data)
        
        f_t = tape.gradient(f, t_data )
        f_y = tape.gradient(f, y_data )
        f_m = tape.gradient(f, mu_data)
        # f_l = tape.gradient(f, lam_data)
        
        f_yy = tape.gradient(f_y, y_data )
        f_ym = tape.gradient(f_y, mu_data)
        f_mm = tf.stack([tape.gradient(f_m[:,i:i+1], mu_data) for i in range(self.config.d)], 1)
                        
        L = self.L(t_data, y_data, mu_data, f_t, f_y, f_m, f_yy, f_ym, f_mm)
        loss =    tf.reduce_mean( (
            tf.square(L)
            # + penalise_range(f_yy, 1e-4, np.inf)
            # + penalise_range(f_l, 1e-4, np.inf)
            ) )
        
        return loss
    
    def terminal_loss(self, data, tape):
        raise NotImplementedError()
        
    def call(self, sample_data, training, tape):
        collocation_data = tf.convert_to_tensor(sample_data['collocation'], dtype=tf.float64)
        boundary_data = tf.convert_to_tensor(sample_data['boundary'], dtype=tf.float64)
        
        collocation_loss = self.collocation_loss(collocation_data, tape)
        terminal_loss = self.terminal_loss(boundary_data, tape)   
        loss = collocation_loss + terminal_loss
                
        return loss


        
            
         
class ValueModel(PINNModel):
        
    def terminal_loss(self, data, tape):
        t_data   = data[:, :1]
        lam_data = data[:,1:2]
        y_data   = data[:,2:3]
        mu_data  = data[:,3: ]
                          
        tape.watch(y_data )
        tape.watch(mu_data)
        
        tracked_data = tf.concat([t_data, lam_data, y_data, mu_data], 1)  
        
        v = self.net(tracked_data)
        # v_y = tape.gradient(v, y_data )
        # v_m = tape.gradient(v, mu_data )
        
        loss =   tf.reduce_mean( (
            tf.square(v - self.config.V(y_data, lam_data))
            # + tf.square(v_y - self.config.V_y(y_data, lam_data))
            # + tf.square(v_m)
            ))
        
        return loss
    
        


class ConstraintModel(PINNModel):
        
    def terminal_loss(self, data, tape):
        t_data   = data[:, :1]
        lam_data = data[:,1:2]
        y_data   = data[:,2:3]
        mu_data  = data[:,3: ]
                          
        tape.watch(y_data )
        tape.watch(mu_data)
        
        tracked_data = tf.concat([t_data, lam_data, y_data, mu_data], 1)  
        
        h = self.net(tracked_data)
        # h_m = tape.gradient(h, mu_data )

        loss =   tf.reduce_mean( (
            tf.square(h - self.config.constraint(y_data, lam_data))
            # + tf.square(h_m)
            ))
        
        return loss    
    
def main():
    
    config = Config()
    tf.keras.backend.clear_session()
    tf.keras.backend.set_floatx('float64')
    
    solver = PINNSolver(config)
    data, train_data = solver.train()

        
        
        
        
    
if __name__ == '__main__':
    main()
    tf.keras.backend.clear_session()

    
    
    


    


