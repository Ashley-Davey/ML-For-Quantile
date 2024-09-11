import tensorflow as tf
import numpy as np
from common import log, b_dot, b_mult, Config
import time
from matplotlib import pyplot as plt, rcParams


rng = np.random.default_rng()
tf .get_logger().setLevel('ERROR') #remove warning messages for taking second derivatives
rcParams['figure.dpi'] = 600


 
#solver

class DualSolver(object):
    def __init__(self, config):
        log('Running Dual Simulation Algorithm')
        self.start_time = time.time()
        self.config = config
        self.model = DualModel(config)
        self.optimiser = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate) 
        self.n = len(self.config.lams)

        
    def get_sample(self, num_sample):
        paths = np.random.normal(size=[num_sample, self.config.d, self.config.dual_time_steps]) 
        dw_sample = paths * self.config.dual_sqrt_delta_t
        return dw_sample

    
    def train(self):
        
        # begin sgd iteration
        
        data = {
            'value': [],
            'loss': [],
            'constraint': [],
            'times': [],
            'y': [],
            }
        
        try:
            try:
                step=0  
                loss = np.NAN
                log(f"Step: {0:6d} \t Time: {time.time() - self.start_time:5.2f} ")
                for step in range(1, self.config.iteration_steps + 1):
                    display = step % self.config.display_step == 0 or step == 1
                    train_data = self.train_step()
                    value = train_data['value'].numpy()
                    loss = train_data['loss'].numpy()
                    constraint = train_data['constraint'].numpy()
                    y = train_data['y'].numpy()
                    data['value'].append(value)
                    data['loss'].append(loss)
                    data['constraint'].append(constraint)
                    data['times' ].append(time.time() - self.start_time)
                    data['y' ].append(y)
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
        tf.keras.backend.clear_session()
        del self.model
        return data, train_data


    @tf.function
    def train_step(self):
        train_data, grads = self.model(self.get_sample(self.config.dual_batch_size))
        self.optimiser.apply_gradients(zip(grads, self.model.trainable_variables))
        return train_data
    
         
class DualModel(tf.keras.Model):
    def __init__(self, config):
        super(DualModel, self).__init__()
        self.config = config
        self.n = len(config.lams)
        y0 = np.random.uniform(1.9 / (config.x0 + 1e-1), 2.1 / (config.x0 + 1e-1), self.n)
        self.y = tf.Variable( y0, dtype = tf.float64)
        self.lams = tf.Variable(self.config.lams, dtype = tf.float64, trainable = False)

    def call(self, sample_data):
        dw = sample_data
        ones_vec = tf.ones(shape=tf.stack([tf.shape(dw)[0], 1]), dtype=tf.float64)
        ones_mat = tf.ones(shape=tf.stack([tf.shape(dw)[0], 1, 1]), dtype=tf.float64)
        ones_d = tf.ones(shape=tf.stack([tf.shape(dw)[0], self.config.d]), dtype=tf.float64)
        lam_vec = ones_vec @ self.lams[np.newaxis, :] # M x n
        
        time_array = tf.range(self.config.dual_time_steps + 1, dtype = tf.float64) * self.config.dual_delta_t
        time_stamp_rev = tf.ones(shape=tf.stack([tf.shape(dw)[0], self.config.dual_time_steps + 1]), dtype=tf.float64) * time_array 
        time_stamp = tf.transpose(time_stamp_rev)[:,:,np.newaxis]
        sigma_inv = self.config.sigma_inv * ones_mat
        Y = ones_vec * 1.0 # M x 1
        mu = ones_d * self.config.mu0 # M x d
        
        for i in range(self.config.dual_time_steps):
            
            Y = Y * (
                1 
                - self.config.r * self.config.dual_delta_t
                - b_dot(b_mult(sigma_inv, (mu - self.config.r)), dw[:,:,i])
                )
                        
            mu = mu + b_mult(self.config.psi(time_stamp[i], mu), dw[:,:,i])
            if self.config.dist == 'discrete':
                mu = tf.minimum(tf.maximum(mu, self.config.mu_l), self.config.mu_h)
                
                
        X = - self.config.V_y(Y @ self.y[np.newaxis, :], lam_vec)
        
        
        # value = tf.reduce_mean(self.config.V(Y @ self.y[np.newaxis, :], lam_vec), axis = 0) + self.config.x0 * self.y
        value = tf.reduce_mean(self.config.U_conc(X, lam_vec), axis = 0)
            
        
            
        train_data  =  { 
            'loss': tf.reduce_mean(value),
            'value': value,
            # 'constraint': 1.0 - tf.reduce_mean(self.config.constraint(Y @ self.y[np.newaxis, :], lam_vec), axis = 0),
            'constraint': 1.0 - tf.reduce_mean(self.config.constraint_primal(X), axis = 0),
            'y': self.y,
            'X': - self.config.V_y(Y @ self.y[np.newaxis, :], lam_vec),
            'Y': Y @ self.y[np.newaxis, :]
            }
        
        grads =  [tf.reduce_mean(Y * self.config.V_y(Y @ self.y[np.newaxis, :], lam_vec), axis = 0) + self.config.x0]
        
        return train_data, grads
                    
    
    
def main():
    
    
        
    config = Config()
    tf.keras.backend.clear_session()
    tf.keras.backend.set_floatx('float64')
    data, train_data = DualSolver(config).train()

    

    
if __name__ == '__main__':
    main()
    tf.keras.backend.clear_session()

    
    
    

         
