"""
urnn_pmnist_test.py

Test of URNN, RNN and LSTMs on permuted MNIST
"""
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, CuDNNLSTM, RNN
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from complex_layers import ComplexRNNCell, get_complex_weights, set_complex_weights

    
def permute_data(Xtr0, Xts0):
    """
    Permutes and reshapes the MNIST data.
    
    To put in the format for an RNN input we reshape the data to    
    `(*,npix,1)` where `npix=784` is the number of pixels per image.
    The pixels are randomly permuted.
    """    
    ntr, nrow, ncol = Xtr0.shape
    nts = Xts0.shape[0]
    npix = nrow*ncol
            
    # Flatten the pixels
    Xtr = Xtr0.reshape((ntr,npix,1))/255.0
    Xts = Xts0.reshape((nts,npix,1))/255.0
    
    # Randomly permute -- seed taken from 
    # https://github.com/stwisdom/urnn/blob/master/mnist.py
    rng_permute = np.random.RandomState(92916)
    I = rng_permute.permutation(npix)
    #I = np.random.permutation(npix)
    Xtr = Xtr[:,I,:]
    Xts = Xts[:,I,:]
           
    return Xtr, Xts
            
            
class ProjectCB(tf.keras.callbacks.Callback):
    """
    Callback to project kernel matrix.
    
    This funciton will be called at the end of each batch and performs
    the projection.
    
    If unitary==True:   matrix is projected to a unitary matrix
    elif contractive:   matrix is projected to a contraction
    else:               no projection
    
    For large networks, this may slow down the overall optimization.
    """
    def __init__(self,rnn_layer,unitary=False,is_complex=False,\
                 contractive=False):
        self.rnn_layer = rnn_layer     
        self.unitary = unitary
        self.is_complex = is_complex
        self.contractive = contractive

    def on_batch_end(self, batch, logs={}):
        # Return if no condition is to be imposed
        if not (self.contractive or self.unitary):
            return
        
        if self.is_complex:
            Wx, Wh, b = get_complex_weights(self.rnn_layer)
        else:
            Wx, Wh, b = self.rnn_layer.get_weights()
            
        U,s,Vtr = np.linalg.svd(Wh)
        if self.unitary:
            Wh = U.dot(Vtr)
        elif self.contractive:
            s = np.minimum(1, s)
            Wh = (U*s[None,:]).dot(Vtr)            
            
        # Reset the weights
        if self.is_complex:
            b = np.minimum(0, b)  # Keep negative
            set_complex_weights(self.rnn_layer, [Wx,Wh,b])
        else:
            self.rnn_layer.set_weights([Wx,Wh,b])         
    
    
class RNNModel:
    def __init__(self, nt = 784, nin = 1, nh = 64, nout = 10,\
                lr=0.001, mod_type='rnn', batch_size=64,\
                is_complex=False, contractive=False):
        """
        A RNN network model
                
        nt:     num time steps per sample
        nin:    num inputs per time step
        nout:   num outputs per time step
        nh:     num hidden units
        lr:           Learning rate
        mod_type:    'rnn', 'urnn' or 'lstm'
        batch_size:   Batch size for the optimization
        is_complex:   If complex is to be performed
            applies only to 'rnn' or 'urnn'.  
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
        self.is_complex = is_complex
        self.contractive = contractive
        
        # Create the model
        self.create_model()
        
        
    def create_model(self):
        """
        Creates the model
        """
        self.mod = Sequential()
        
        # Add the RNN layer
        # Note 1:  For RNN, we set unroll=True to enable fast GPU usage
        # Note 2:  You need to set the CuDNN version of LSTM
        unroll = tf.test.is_gpu_available()       
        if (self.mod_type == 'lstm'):
            # LSTM model
            if tf.test.is_gpu_available():
                self.mod.add(CuDNNLSTM(self.nh, input_shape=(self.nt, self.nin),\
                                  return_sequences=False, name='RNN'))
            else:
                self.mod.add(LSTM(self.nh, input_shape=(self.nt, self.nin),\
                                  return_sequences=False, name='RNN',unroll=unroll))
                
        elif self.is_complex:
            # Complex RNN
            cell = ComplexRNNCell(nh=self.nh)
            self.mod.add(RNN(cell, input_shape=(self.nt, self.nin),\
                return_sequences=False, name='RNN',unroll=True))
        else:
            # Real RNN model                
            self.mod.add(SimpleRNN(self.nh, input_shape=(self.nt, self.nin),\
                return_sequences=False, name='RNN',activation='relu',unroll=unroll))
        self.mod.add(Dense(nout,activation='softmax',name='Output'))
        self.mod.summary()
        
        

    def fit(self, Xtr,Ytr,Xts,Yts,nepochs=10):
        """
        Fits the model parameters
        """
        # Compile the model
        #opt = Adam(lr=self.lr)
        opt = RMSprop(lr=self.lr)
        self.mod.compile(loss='sparse_categorical_crossentropy', optimizer=opt,\
                         metrics=['accuracy'])         
        
        # For URNN, add a callback that projects the weight matrix to unitary
        if mod_type == 'urnn':
            rnn_layer = self.mod.get_layer('RNN')
            callbacks = [ProjectCB(rnn_layer,unitary=True,is_complex=self.is_complex)]
        elif mod_type == 'rnn':
            rnn_layer = self.mod.get_layer('RNN')
            callbacks = [ProjectCB(rnn_layer,unitary=False,is_complex=self.is_complex,\
                                   contractive=contractive)]
        else:
            callbacks = []
        
        # Print progress
        if self.mod_type == 'lstm':
            cstr = ''
        elif self.is_complex:
            cstr = ' (complex)'
        else:
            cstr = ' (real)'
        print('%s%s nh=%d' % (self.mod_type, cstr, self.nh))          
        
        # Fit the model
        hist = self.mod.fit(Xtr,Ytr,epochs=nepochs, batch_size=self.batch_size,\
                            callbacks=callbacks,validation_data=(Xts,Yts))        
        self.tr_acc  = hist.history['acc']
        self.val_acc = hist.history['val_acc']
        


if __name__ == "__main__":   

    
    """
    Parse arguments from command line
    """
    parser = argparse.ArgumentParser(description='Permuted MNIST RNN test')
    parser.add_argument('--nepochs',action='store',default=100,type=int,\
        help='number of epochs for each model')
    parser.add_argument('--lr',action='store',default=0.0001,type=float,\
        help='learning rate')
    parser.add_argument('--nh',action='store',nargs='+',\
        default=[16,32,48,64,80,96],type=int,
        help='num hidden units')
    parser.add_argument('--mod_type', action='store', default='rnn',\
        help='model type (rnn, urnn, lstm)')    
    parser.add_argument('--fn_prefix', action='store', default='pmnist_results',\
        help='filename prefix to store results')
    parser.add_argument('--batch_ind',action='store',\
        default=4,type=int,\
        help='batch index for array processing.  -1=do all runs')
    parser.add_argument('--complex', dest='is_complex', action='store_true',\
        help="Uses complex RNN or URNN")
    parser.set_defaults(is_complex=False)    
    parser.add_argument('--contractive', dest='contractive', action='store_true',\
        help="For RNN, forces the transition matrix to be contractive")    
    parser.set_defaults(contractive=False)    
    
    # Parse default args            
    args = parser.parse_args()
    batch_ind = args.batch_ind
    
    # Get the file name
    fn_pre = args.fn_prefix
    if batch_ind >= 0:
        fn = '%s%d.p' % (fn_pre,batch_ind)
    else:
        fn = fn_pre    
    print('filename = %s' % fn)
     
    """
    Main simulation
    """
   
    # Initialize the data
    tr_acc = []
    val_acc  = []
    
    # Get data
    mnist = tf.keras.datasets.mnist
    (Xtr0, Ytr),(Xts0, Yts) = mnist.load_data()
    Xtr,Xts = permute_data(Xtr0,Xts0)
    
    # Get dimensions
    nin = 1
    nout = 10
    nt = Xtr.shape[1]
    
    test_param = [\
                  ['test_name', 'crnn', 'mod_type', 'rnn', 'contractive', True,\
                   'nh', [16,32,48,64]],\
                  ['test_name', 'rnn', 'mod_type', 'rnn',\
                   'contractive', False, 'nh', [16,32,48,64]],\
                  ['test_name', 'urnn', 'mod_type', 'urnn','nh', [16,32,48,64]],\
                  ['test_name', 'crnn', 'mod_type', 'rnn', 'contractive', True,\
                   'nh', [80,96]],\
                  ['test_name', 'rnn', 'mod_type', 'rnn',\
                   'contractive', False, 'nh', [80,96]],\
                  ['test_name', 'urnn', 'mod_type', 'urnn','nh', [80,96]],\
                  ['test_name', 'crnn', 'mod_type', 'rnn', 'contractive', True,\
                   'nh', [128]],\
                  ['test_name', 'rnn', 'mod_type', 'rnn',\
                   'contractive', False, 'nh', [128]],\
                  ['test_name', 'urnn', 'mod_type', 'urnn','nh', [128]]\
                  ]
                   
                   
    ntest = len(test_param)
    
    # Get tests to run
    # If batch_ind was selected only run the specified test
    # Otherwise, run all tests
    if batch_ind >= 0:
        tests = [np.mod(batch_ind,ntest)]
    else:
        tests = np.arange(ntest)
        
    
    # Loop over model types
    for it in tests:
    
        # Get default arguments
        nepochs = args.nepochs
        nh_list  = args.nh
        lr = args.lr
        is_complex = args.is_complex
        contractive = args.contractive
        test_name = 'test %d' % it
        mod_type = 'urnn'
        
        # Get parameters specific to the test
        param = test_param[it]
        nparam = len(param)//2
        for i in range(nparam):
            pname = param[2*i]
            pval = param[2*i+1]
            if pname == 'test_name':
                test_name = pval
            elif pname == 'mod_type':
                mod_type = pval
            elif pname == 'contractive':
                contractive = pval
            elif pname == 'lr':
                lr = pval
            elif pname == 'nh':
                nh_list = pval
            else:
                raise ValueError('Unknown parameter %s' % pname)
        print('%s, mod_type=%s, cont=%s, lr=%f' % \
              (test_name,mod_type,str(contractive),lr) )
                
        # Loop over number of hidden units in model
        for nh in nh_list:
            
            print('nh=%d ' % nh)
                                    
            # Create a RNN model
            K.clear_session()
            mod_est = RNNModel(nt=nt,nin=nin,nh=nh,nout=nout,mod_type=mod_type,\
                               lr=lr,is_complex=is_complex,\
                               contractive=contractive)                
            # Fit the model
            mod_est.fit(Xtr,Ytr,Xts,Yts,nepochs=nepochs)
            
            # Save the validation loss
            tr_acc.append(mod_est.tr_acc)
            val_acc.append(mod_est.val_acc)
    
    
    """ 
    Save results
    """    
    version = 1
    import pickle
    with open(fn, 'wb') as fp:
        pickle.dump({'nh': args.nh, 'is_complex': args.is_complex, 'lr': args.lr,\
                     'test_param': test_param, 'batch_ind': batch_ind,\
                     'val_acc': val_acc, 'tr_acc': tr_acc}, fp)
    
