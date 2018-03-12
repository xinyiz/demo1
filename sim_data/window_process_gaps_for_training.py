import math
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
from enum import Enum
class TrainingDataProcessor(object):
    def __init__(self,num_features,num_keys,gap_size,in_length):
        self.num_features=num_features # without the additional missing token
        self.keys_features=[1]*(num_features+1)
        self.target_features=[1]*(num_features+1)
        self.num_keys = num_keys
        self.gap_size = gap_size
        self.gap_value = 0
        self.in_length = in_length
        self.data_length = self.num_keys+(self.num_keys-1)*self.gap_size
    def set_keys_features(self,keys_feature_bools):
        assert(len(keys_feature_bools) == self.num_features)
        self.keys_features = keys_feature_bools + [True]
    def set_target_features(self,target_feature_bools):
        assert(len(target_feature_bools) == self.num_features)
        self.target_features = target_feature_bools + [True]
    def make_interleaved(self,data,int_seq_id,int_seq_starts):
        result = np.zeros((self.data_length,self.num_features+1)) # add one dim for missing token
        # make gaps
        for k in range(self.num_keys - 1):
            ki = k*(self.gap_size+1)+1
            result[ki:ki+self.gap_size,:] = self.gap_value # make a gap
            result[ki:ki+self.gap_size,self.num_features] = 0 # set missing token to 0 
        # make keys
        seq_i = np.random.randint(0,len(int_seq_id),args.num_keys)
        for k in range(self.num_keys):
            sele = data[int_seq_id[seq_i[k]],int_seq_starts[seq_i[k]]+k*(self.gap_size+1),:] 
            result[k*(self.gap_size+1),:] = sele
            result[k*(self.gap_size+1),self.num_features] = 1 
        return result
    def convert_keys(self,in_data,s_idx):
        # select the features we want for keyframes
        result = in_data[s_idx:s_idx+self.data_length,self.keys_features]
        # make keys
        for k in range(self.num_keys):
            result[k*(self.gap_size+1),self.num_features] = 1 
        # make gaps
        for k in range(self.num_keys - 1):
            ki = k*(self.gap_size+1)+1
            result[ki:ki+self.gap_size,:] = self.gap_value # make a gap
            result[ki:ki+self.gap_size,self.num_features] = 0 # set gap features to 0 
        return result
    def convert_target(self,in_data,s_idx):
        # select the features we want for keyframes
        result = in_data[s_idx:s_idx+self.data_length,self.target_features]
        # set all frames to not missing
        result[:,self.num_features] = 1 
        return result

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--num_samples", action="store", type=int, default=40000)
    parser.add_argument("--dir_data", action="store",  default=".")

    parser.add_argument("--num_keys", action="store", type=int, default=3)
    parser.add_argument("--gap_size", action="store", type=int, default=15)
    parser.add_argument("--dims", action="store", type=int, default=10)

    parser.add_argument("--in_feature_bools", action="store",default="1 1 1 1 1 1 1 1 1 1") 
    parser.add_argument("--target_feature_bools", action="store",default="1 1 1 1 1 1 1 1 1 1")
    parser.add_argument("--num_interleaved_seq", action="store",default=10,type=int)
    parser.add_argument("--num_seqs", action="store",default=3,type=int)

    args = parser.parse_args()
    data = torch.load('%s/particle_spring_40000_10.pt'%args.dir_data, 'wb')
    T = int(data.shape[1]/args.dims)
    data = data[0:args.num_samples,:]

    # reshape and add the missing token feature dim
    shaped = data.reshape(args.num_samples,T,args.dims).copy()
    shaped_data = np.zeros((data.shape[0],T,args.dims+1)) 
    shaped_data[:,:,:-1] = shaped 

    total_data_size = args.num_keys+(args.num_keys-1)*args.gap_size
    data_keys = np.zeros((shaped_data.shape[0],total_data_size,args.dims+1)) 
    data_target = np.zeros((shaped_data.shape[0],total_data_size,args.dims+1)) 

    processor = TrainingDataProcessor(args.dims,args.num_keys,args.gap_size,T)
    processor.set_keys_features([bool(int(v)) for v in args.in_feature_bools.split()])
    processor.set_target_features([bool(int(v)) for v in args.target_feature_bools.split()])

    key_starts = np.random.randint(0,T-total_data_size-1,args.num_samples)
    for i in range(args.num_samples):
        data_keys[i] = processor.convert_keys(shaped_data[i],key_starts[i])
        data_target[i] = processor.convert_target(shaped_data[i],key_starts[i])

    # Make artificial key frame data
    data_interleaved = np.zeros((args.num_interleaved_seq,total_data_size,args.dims+1)) 
    int_seq_id = []
    int_seq_starts = []
    for k in range(args.num_interleaved_seq):
        int_seq_id += [np.random.randint(0,args.num_samples,args.num_seqs)]
        int_seq_starts += [np.random.randint(0,T-total_data_size-1,args.num_seqs)]

    for i in range(args.num_interleaved_seq):
        data_interleaved[i] = processor.make_interleaved(shaped_data,int_seq_id[i],int_seq_starts[i])

    torch.save(data_keys, open('particle_spring_%d_%d_keys.pt'%(args.num_samples,args.gap_size), 'wb'))
    torch.save(data_target, open('particle_spring_%d_%d_target.pt'%(args.num_samples,args.gap_size), 'wb'))
    torch.save(data_interleaved, open('particle_spring_%d_%d_interleaved.pt'%(args.num_samples,args.gap_size), 'wb'))


    data_keys = data_keys.reshape(args.num_samples,total_data_size*(args.dims+1))
    data_target = data_target.reshape(args.num_samples,total_data_size*(args.dims+1))
    data_interleaved = data_interleaved.reshape(args.num_interleaved_seq,total_data_size*(args.dims+1))
    if args.plot:
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Input data with gaps', fontsize=30) 
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.axis([0, total_data_size,-30.0,30.0])
        print("dims",args.dims+1)
        def draw(i,color):
            plt.plot(np.repeat(np.arange(total_data_size),args.dims+1), data_keys[i,:], 'o',markerfacecolor='None',markeredgecolor=color,ms=50)
            plt.plot(np.repeat(np.arange(total_data_size),args.dims+1), data_target[i,:], 'o',c='b')
            plt.savefig('gaps_%d.pdf'%i)
            plt.clf()
            plt.plot(np.repeat(np.arange(total_data_size),args.dims+1), data_interleaved[i,:],'^', 
                     markerfacecolor='None', 
                     markeredgecolor='m')
            plt.savefig('artificial_data_%d.pdf'%i)
            plt.clf()
        for j in range(0,2):
            print("plotting sample input %d:"%j)
            draw(j,'r')

