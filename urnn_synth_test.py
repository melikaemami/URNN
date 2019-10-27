"""
urnn_synth_test.py

Test of URNN and RNN model fitting on synthetic data
"""
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import SimpleRNN, LSTM, Input, TimeDistributed, Dense
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='Loads images from flickr')
parser.add_argument('--ntr',action='store',default=700,type=int,\
    help='number of training samples')
parser.add_argument('--nts',action='store',default=300,type=int,\
    help='number of test samples')
parser.add_argument('--nin',action='store',default=2,type=int,\
    help='number of inputs')
parser.add_argument('--nh',action='store',default=4,type=int,\
    help='number of hidden states')
parser.add_argument('--nt',action='store',default=1000,type=int,\
    help='number of time steps per trial')
parser.add_argument('--nout',action='store',default=2,type=int,\
    help='number of outputs')
parser.add_argument('--nepochs',action='store',default=100,type=int,\
    help='number of epochs for each model')
parser.add_argument('--snr',action='store',default=10.0,type=float,\
    help='SNR in dB')
parser.add_argument('--lr',action='store',default=0.001,type=float,\
    help='learning rate')
parser.add_argument('--nh_rnn',action='store',nargs='+',\
    default=[],type=int,\
    help='num hidden units to test for RNN')
parser.add_argument('--nh_urnn',action='store',nargs='+',\
    default=[],type=int,\
    help='num hidden units to test for URNN')
parser.add_argument('--nh_lstm',action='store',nargs='+',\
    default=[4],type=int,\
    help='num hidden units to test for URNN')
parser.add_argument('--fn_prefix', action='store', default='urnn_results',\
    help='filename prefix to store results')
parser.add_argument('--fn_suffix',action='store',\
    default=0,type=int,\
    help='filename suffix')
parser.add_argument('--bias_adj', dest='bias_adj', action='store_true',\
    help="Performs automatic bias adjustment")
parser.set_defaults(bias_adj=True)

    
args = parser.parse_args()
ntr  = args.ntr
nts  = args.nts
nin  = args.nin
nt  = args.nt
nout  = args.nout
nh  = args.nh
snr = args.snr
nepochs = args.nepochs
nh_rnn  = args.nh_rnn
nh_urnn  = args.nh_urnn
nh_lstm  = args.nh_lstm
bias_adj = args.bias_adj
lr = args.lr

# Get the file name
fn_pre = args.fn_prefix
fn_suf = args.fn_suffix
fn = '%s%d.p' % (fn_pre,fn_suf)

nimage = ntr + nts
print("num training  %d " % ntr)
print("num test      %d " % nts)
print("nh  =%d " % nh)
print("nt  =%d " % nt)
print("nin =%d " % nin)
print("nout=%d " % nout)
print('filename = %s' % fn)


import tqdm
class SyntheticRNN:
    def __init__(self, nt = 100, nin = 1, nh = 4, nout = 2,\
                sparse_in=1.0, rho_w=0.01, sparse_tgt = 0.6, snr=20,\
                bias_adj=False):
        """
        A random synthetic RNN
        
        sparse_in:   Fraction of input that is sparse
        snr:         SNR in dB
        rho_w:       Eigenvalues of W will be in [1-rho_w,1]
        bias_adj:    Biases will be adjusted for a target sparsity
        sparse_tgt:  Sparsity target when bias adjustment is enabled
        """
        # Save dimensions
        self.nt = nt
        self.nin = nin
        self.nh = nh
        self.nout = nout
        
        # Save other parameters
        self.sparse_in = sparse_in
        self.snr = snr
        self.rho_w = rho_w 
        self.sparse_tgt = sparse_tgt 
        self.bias_adj = bias_adj
        
        # Create the model
        self.create_model()
        self.gen_rand_weights()
        if self.bias_adj:
            self.adjust_biases()
        
        
    def create_model(self):
        """
        Creates the model
        """
        x_op = Input(shape=(self.nt, self.nin), name='x')
        h_op = SimpleRNN(self.nh, return_sequences=True, name='RNN',activation='relu')(x_op) 
        y_op = TimeDistributed(Dense(self.nout,activation='linear'),name='Output')(h_op) 
        self.mod = Model(x_op, [h_op,y_op])
        
    def gen_rand_weights(self):
        """
        Set the weights of the model for a given sparisty and eigenvalue constraint
        """
        # Generate random matrix Whid
        A = np.random.normal(0,1,(self.nh,self.nh))
        Whid = np.eye(self.nh) - 0.01*A.T.dot(A)

        # Generate the other hidden layer parameters        
        Wxin = np.random.normal(0,1,(self.nin,self.nh))
        if self.bias_adj:
            bhid = np.random.normal(0,1,self.nh)
        else:
            bhid = np.zeros(self.nh)

        # Output layer parameters
        Wout = np.random.normal(0,1,(self.nh,self.nout))        
        bout = np.random.normal(0,1,(self.nout,))
        
        #setting the weight matrices :
        layer = self.mod.get_layer('RNN')
        layer.set_weights([Wxin,Whid,bhid])

        layer = self.mod.get_layer('Output')
        layer.set_weights([Wout,bout])
        
    def adjust_biases(self):
        """
        Adjust the biases such for the sparsity target
        """
        #choose some random input
        ns_cal = 100
        X = np.random.normal(0,1,(ns_cal, self.nt, self.nin))*\
            (np.random.uniform(0,1,(ns_cal, self.nt, self.nin)) < self.sparse_in)
        
        # Get initial weights
        layer = self.mod.get_layer('RNN')
        Wxin,Whid,bhid = layer.get_weights()

        
        # Adjustment loop
        nit = 100
        sparse = np.zeros((nit,self.nh))

        print('Adjusting biases...')
        pbar = tqdm.tqdm(total=nit, ncols=100)
        for it in  range(nit):
            # Predict the ouptut
            H, Y0 = self.mod.predict(X)
            
            # Measure the sparsity
            sparse[it,:] = np.mean(H>1e-3, axis=(0,1))
            sparse_err = sparse[it,:]-self.sparse_tgt

            # Adjust the biases
            bhid = bhid - 0.1*sparse_err
            layer.set_weights([Wxin,Whid,bhid])
            pbar.update(1)
        
        pbar.close()
            
        #if np.any(np.abs(sparse_err) > 0.05):
        #    raise ValueError('Sparsity target does not converge')
        #print('Converged')
    
        #plt.plot(sparse)
        
    def gen_data(self, ntr=700, nts=300):
        """
        Generate training and test data
        """
        
        # Generate data
        ns = ntr+nts
        X = np.random.normal(0,1,(ns, self.nt, self.nin))*\
            (np.random.uniform(0,1,(ns, self.nt, self.nin)) < self.sparse_in)
        H, Y0 = self.mod.predict(X)
        
        # Add noise
        wvar = np.mean(Y0**2)*10**(-0.1*self.snr)
        W = np.random.normal(0,np.sqrt(wvar),(ns,self.nt,self.nout))
        Y = Y0 + W
        
        # Split into training and test
        Xtr = X[:ntr]
        Ytr = Y[:ntr]
        Xts = X[ntr:]
        Yts = Y[ntr:]
        
        return Xtr,Ytr, Xts,Yts, wvar
    
def project(rnn_layer):
    """
    Projects the weights of an RNN to a unitary matrix
    """
    Wx, Wh, b = rnn_layer.get_weights()
    U,s,Vtr = np.linalg.svd(Wh)
    Wh = U.dot(Vtr)
    rnn_layer.set_weights([Wx,Wh,b])  
    
class RNNModel:
    def __init__(self, nt = 100, nin = 1, nh = 4, nout = 2,\
                lr=0.001, mod_type='rnn', batch_size=10):
        """
        A RNN network model
        
        lr:          Learning rate
        unitary:     Impose unitary constraint (i.e. makes it a URNN)
        """
        # Save dimensions
        self.nt = nt
        self.nin = nin
        self.nh = nh
        self.nout = nout
        
        # Save parameters
        self.batch_size=batch_size
        self.mod_type = mod_type
        self.lr = lr
        
        # Create the model
        self.create_model()
        
        # Initialize the training history
        self.val_loss = []
        
        
    def create_model(self):
        """
        Creates the model
        """
        x_op = Input(shape=(self.nt, self.nin), name='x')
        if (self.mod_type == 'lstm'):
            h_op = LSTM(self.nh, return_sequences=True, name='RNN')(x_op) 
        else:
            h_op = SimpleRNN(self.nh, return_sequences=True, name='RNN',activation='relu')(x_op) 
        y_op = TimeDistributed(Dense(nout,activation='linear'),name='Output')(h_op) 
        self.mod = Model(x_op, y_op)
        

    def fit(self, Xtr,Ytr,Xts,Yts,nepochs=10):
        """
        Fits the model parameters
        """
        # Compile the model
        opt = Adam(lr=self.lr)
        self.mod.compile(loss='mean_squared_error', optimizer=opt)
        
        # Get training parameters
        ntr = Xtr.shape[0]
        batch_size = self.batch_size
        nsteps = ntr // batch_size

        # Open a progress bar
        desc = '%4s nh=%d' % (self.mod_type, self.nh)
        pbar = tqdm.tqdm(desc=desc, total=nepochs, ncols=100)

        # Loop over epochs
        rnn_layer = self.mod.get_layer('RNN')
        for iepoch in range(nepochs):
            # Shuffle training data
            I = np.random.permutation(ntr)
            for i in range(nsteps):
                Ib = I[i*batch_size:(i+1)*batch_size]
                Xb = Xtr[Ib]
                yb = Ytr[Ib]
                self.mod.train_on_batch(Xb,yb)
                if mod_type == 'urnn':
                    project(rnn_layer)
                    
            # Evaluate the loss
            v = self.mod.evaluate(Xts,Yts,verbose=0)
            self.val_loss.append(v)
            loss_str = 'Val Loss: %12.4e' % v
            pbar.set_postfix_str(loss_str)
            pbar.update(1)            
            
        pbar.close()    
        
        
        
K.clear_session()

val_loss = []

# Generate data
K.clear_session()
mod_true = SyntheticRNN(nt=nt,nin=nin,nh=nh,nout=nout,snr=snr,\
                        bias_adj=bias_adj)
Xtr,Ytr, Xts,Yts, wvar = mod_true.gen_data(ntr = 700, nts=300)

print('Optimal loss=%12.4e' % wvar)
for mod_type in ['rnn', 'urnn', 'lstm']:
    
    if mod_type == 'rnn':
        nh_mod = nh_rnn
    elif mod_type == 'urnn':
        nh_mod = nh_urnn
    elif mod_type == 'lstm':
        nh_mod = nh_lstm

    for nh1 in nh_mod:                        
        # Create a RNN model
        K.clear_session()
        mod_est = RNNModel(nt=nt,nin=nin,nh=nh1,nout=nout,mod_type=mod_type,\
                           lr=lr)        
        
        # Fit the model
        mod_est.fit(Xtr,Ytr,Xts,Yts,nepochs=nepochs)
        
        # Save the validation loss
        val_loss.append(mod_est.val_loss)


""" 
Save results
"""    
version = 2
import pickle
with open(fn, 'wb') as fp:
    pickle.dump([version, nh_rnn, nh_urnn, nh_lstm, wvar, val_loss,\
                 snr, nin, nout], fp)
    
