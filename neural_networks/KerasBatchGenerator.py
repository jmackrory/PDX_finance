import numpy as np
import keras

class KerasDeepBatchGenerator(keras.utils.Sequence):

    def __init__(self, data):
        """KerasBatchGenerator(data, input_col_ind, output_col_ind,conf)
        
        Generator for use with Keras Batch functions.
        Note:this returns flattened 1D-sequences for deep network.
        """
        self.data=data
        self.input_col_ind=input_col_ind
        self.output_col_ind=output_col_ind
        self.Nbatch=config.Nbatch
        self.Ninput=config.Ninput
        self.Ntime_in=config.Ntime_in
        self.Ntime_out=config.Ntime_out

        
    def generate(self):
        """generate()
        Returns a batch of output.
        Returns a randomly selected batch of input/output sequences.
        Inputs are all stocks, ETFs and indicators
        Outputs are just future ETFs from input sequence endpoint.
        """
        #starting indices
        ind=np.arange(len(X[:,0])-self.Ntime_in-self.Ntime_out)
        rand_ind=np.random.choice(ind,self.Nbatch,replace=False)
        Xb=np.zeros((self.Nbatch,self.Ninput))
        yb=np.zeros((self.Nbatch,self.Noutput))
        #now populate table (couldn't see nice way to vectorize this assignment, mabe via overloading)
        while True:
            for i in range(self.Nbatch):
                t0=rand_ind[i]
                t1=t0+self.Ntime_in
                t2=t1+self.Ntime_out
                #input all past parameters
                Xb[i]=self.data[t0:t1,self.input_col_ind].reshape(-1)
                #target future ETFs
                yb[i]=self.data[t1:t2,self.output_col_ind].reshape(-1)
            yield Xb,yb

class KerasRNNBatchGenerator(keras.utils.Sequence):

    def __init__(self, data):
        """KerasBatchGenerator(data, input_col_ind, output_col_ind,conf)
        
        Generator for use with Keras Batch functions.
        Note:this returns flattened 1D-sequences for deep network.
        """
        self.data=data
        self.input_col_ind=input_col_ind
        self.output_col_ind=output_col_ind
        self.Nbatch=config.Nbatch
        self.Ninput=config.Ninput
        self.Ntime_in=config.Ntime_in
        self.Ntime_out=config.Ntime_out
        
    def generate(self):
        """generate()
        Returns a batch of output.
        Returns a randomly selected batch of input/output sequences.
        Inputs are all stocks, ETFs and indicators
        Outputs are just future ETFs from input sequence endpoint.
        """
        #starting indices
        ind=np.arange(len(X[:,0])-self.Ntime_in-self.Ntime_out)
        rand_ind=np.random.choice(ind,self.Nbatch,replace=False)
        #initialize batches as zero
        Xb=np.zeros((self.Nbatch,self.Ntime_in,self.Ninput))
        yb=np.zeros((self.Nbatch,self.Ntime_out,self.Noutput))
        #now populate table (couldn't see nice way to vectorize this assignment, mabe via overloading)
        while True:
            for i in range(self.Nbatch):
                t0=rand_ind[i]
                t1=t0+self.Ntime_in
                t2=t1+self.Ntime_out
                #input all past parameters
                Xb[i]=self.data[t0:t1,self.input_col_ind]
                #target future ETFs
                yb[i]=self.data[t1:t2,self.output_col_ind]
            yield Xb,yb

