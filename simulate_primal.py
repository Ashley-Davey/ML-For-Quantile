# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:51:12 2024

@author: ashle
"""

import tensorflow as tf
import numpy as np
from common import log, b_dot, b_mult, Config, FeedForwardSubNet, ConstantModel
import time
from matplotlib import pyplot as plt, rcParams


rng = np.random.default_rng()
tf .get_logger().setLevel('ERROR') #remove warning messages for taking second derivatives
rcParams['figure.dpi'] = 600


 
#solver

class PrimalSolver(object):
    def __init__(self, config, pi_model = FeedForwardSubNet):
        log('Running Primal Simulation Algorithm')
        self.start_time = time.time()
        self.config = config
        self.model = PrimalModel(config, pi_model)
        self.optimiser = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate / 10 )

            
    def get_sample(self, num_sample, final = False):
        if final:
            size=[num_sample * self.config.lam_step, self.config.d, self.config.primal_time_steps]
        else:
            size=[num_sample * self.config.primal_points, self.config.d, self.config.primal_time_steps]

        paths = np.random.normal(size = size) 
        dw_sample = paths * self.config.sqrt_delta_t

        if final:
            lam_points  = self.config.lams[:, np.newaxis]  
            x_points  = np.ones_like(lam_points) * self.config.x0
            mu_points  = np.ones_like(lam_points) @ self.config.mu0[np.newaxis, :]  
        else:
            lam_points  = rng.uniform(self.config.lam_range[0], self.config.lam_range[1], (self.config.primal_points, 1))
            x_points  = rng.uniform(self.config.x_range[0], self.config.x_range[1], (self.config.primal_points, 1))
            mu_points  = np.array([rng.uniform(mu_range[0], mu_range[1], self.config.primal_points) for mu_range in self.config.mu_range]).T
            # lam_points  = rng.choice(self.config.lam_sample, size = (self.config.primal_points, 1))
            # lam_points  = np.linspace(self.config.lam_min, self.config.lam_max, self.config.primal_points)[:, np.newaxis]
            # x_points  = rng.choice(self.config.x_sample, size = (self.config.primal_points, 1))
            # mu_points  = np.array([rng.choice(mu_sample, size = self.config.primal_points) for mu_sample in self.config.mu_sample]).T
        return dw_sample, lam_points, x_points, mu_points
    

    
    def train(self):
        
        # begin sgd iteration
        
        data = {
            'value': [],
            'loss': [],
            'constraint': [],
            'times': [],
            }
        self.evaluate()
        self.optimiser.build(self.model.trainable_variables)

        step=0                    
        log(f"Step: {0:6d} \t Time: {time.time() - self.start_time:5.2f} ")
        #initialise
        try:
            try:
                for step in range(1, int(self.config.primal_steps) + 1):
                    display = step % (self.config.display_primal) == 0 or step == 1
                    train_data = self.train_step()
                    value = train_data['value'].numpy()
                    loss = -1 * train_data['loss'].numpy()
                    constraint = train_data['constraint'].numpy()
                    data['value'].append(value)
                    data['loss'].append(loss)
                    data['constraint'].append(constraint)
                    data['times' ].append(time.time() - self.start_time)
                    if display:
                        log(f"Step: {step:6d} \t Time: {time.time() - self.start_time:5.2f} \t  loss: {loss:+.3e}")
                        
            except Exception as e:
                    log('Termination due to ', e)
                    pass
        except BaseException:
            log('Terminated Manually')
            pass
        log(f"Ended at {step} iterations")

        log(f"Step: {step:6d} \t Time: {time.time() - self.start_time:5.2f} \t  loss: {loss:+.3e}")

        try:
            try:
                train_data = self.evaluate()
                value = train_data['value'].numpy()
                loss = -1 * train_data['loss'].numpy()
                constraint = train_data['constraint'].numpy()

            except Exception as e:
                    log('Termination due to ', e)
                    pass
        except BaseException:
            log('Terminated Manually')
            pass

        log(f"Evaluation:  \t Time: {time.time() - self.start_time:5.2f} \t  loss: {loss:+.3e}")
        return data, train_data
    
    def control_func(self, *args, **kwargs):
        return self.model.pi(*args, **kwargs)


    @tf.function
    def train_step(self):

        dw_sample, lam_points, x_points, mu_points = self.get_sample(self.config.batch_size)
        
        with tf.GradientTape() as tape:
            sample_data = (
                # np.tile(dw_sample, [len(lam_points), 1, 1]),
                dw_sample,
                np.repeat(lam_points, self.config.batch_size )[:, np.newaxis],
                np.repeat(x_points, self.config.batch_size)[:, np.newaxis],
                np.repeat(mu_points, self.config.batch_size, axis = 0),
                )
            eval_data, loss = self.model(sample_data, tape)



        grads = tape.gradient(loss, self.model.pi.trainable_variables) 
        self.optimiser.apply_gradients(zip(grads, self.model.pi.trainable_variables))
        del tape


        return eval_data
    
    
    def evaluate(self):

        dw_sample, lam_points, x_points, mu_points = self.get_sample(self.config.simulation_size, final = True)
        ones = np.ones((self.config.simulation_size, 1))
        n = len(lam_points)
        for i, lam, x, mu in zip(range(n), lam_points, x_points, mu_points):
            sample_data = (
                dw_sample[i * self.config.simulation_size: (i + 1) * self.config.simulation_size ],
                ones * lam,
                ones * x,
                ones @ mu[:, np.newaxis]
                )
            with tf.GradientTape() as tape:
                train_data, loss = self.model(sample_data, tape)
            del tape

            
            if i == 0:
                eval_data = train_data
            else:
                eval_data['loss'] += train_data['loss']

                for key in ['value', 'constraint']:
                    eval_data[key] = tf.concat([eval_data[key], train_data[key]], axis = 0)
                for key in ['X', 'X_old']:
                    eval_data[key] = tf.concat([eval_data[key], train_data[key]], axis = 1)
                    
            eval_data['loss'] = eval_data['loss'] / n

        return eval_data

             
class PrimalModel(tf.keras.Model):
    def __init__(self, config, pi_model):
        super(PrimalModel, self).__init__()
        self.config = config
        self.d = self.config.d
        self.lam = tf.Variable(self.config.lams, dtype = tf.float64, trainable = False)
        # self.pi = pi_model(dim = self.d * self.n)
        self.pi = pi_model([config.primal_hidden, config.primal_hidden], dim = self.d)

    def simulate(self, dw, lam, x, mu, tape):
        ones_mat = tf.ones(shape=tf.stack([tf.shape(dw)[0], 1, 1]), dtype=tf.float64)
        # ones_d = tf.ones(shape=tf.stack([tf.shape(dw)[0], self.d]), dtype=tf.float64)
        # ones_vec = tf.ones(shape=tf.stack([tf.shape(dw)[0], 1]), dtype=tf.float64)
        time_array = tf.range(self.config.primal_time_steps + 1, dtype = tf.float64) * self.config.delta_t
        time_stamp_rev = tf.ones(shape=tf.stack([tf.shape(dw)[0], self.config.primal_time_steps + 1]), dtype=tf.float64) * time_array 
        time_stamp = tf.transpose(time_stamp_rev)[:,:,np.newaxis]
        
        sigma = self.config.sigma * ones_mat # M x d x d 
        X = tf.cast(x, tf.float64) # M x 1
        min_X = X
        mu =  tf.cast(mu, tf.float64) # M x d
        for i in range(self.config.primal_time_steps):
            with tape.stop_recording():
                state = tf.concat([time_stamp[i], lam, X, mu], 1)
            pi = self.pi(state) # M x d
            drift  = (self.config.r + b_dot(pi, mu - self.config.r)) # M x 1
            diffusion = b_mult(sigma, pi ) # M x d
            X = X * (1  + drift * self.config.delta_t + b_dot(diffusion, dw[:,:,i]))
            min_X = tf.minimum(min_X, X)
            mu = mu + b_mult(self.config.psi(time_stamp[i], mu), dw[:,:,i])
            if self.config.dist == 'discrete':
                mu = tf.minimum(tf.maximum(mu, self.config.mu_l), self.config.mu_h)
        X_conc = self.config.concavify(X, lam)
        return X, min_X, X_conc

        
        

    def call(self, sample_data, tape ):
        dw, lam, x, mu = sample_data
        X, min_X, X_conc = self.simulate(dw, lam, x, mu, tape)
        
        loss  = (
            - tf.reduce_mean(self.config.U_conc(X, lam) ) 
            - tf.reduce_mean( tf.square(tf.minimum(min_X, 0))) 
            )

        value =  tf.reduce_mean(self.config.U_conc(X_conc, lam), axis = 0 )
        constraint = 1 - tf.reduce_mean(self.config.constraint_primal(X_conc), axis = 0 )

        train_data  =  { 
            'loss': loss,
            'value': value,
            'constraint': constraint,
            'X_old': X,
            'X': X_conc
            }
        
        
        return train_data, loss
                    
    
    
def main():
    config = Config()
    tf.keras.backend.clear_session()
    tf.keras.backend.set_floatx('float64')
    data, train_data = PrimalSolver(config).train()
        
if __name__ == '__main__':
    main()
    tf.keras.backend.clear_session()

    
    
    

         
