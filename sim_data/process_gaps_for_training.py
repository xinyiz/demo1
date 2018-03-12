import math
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
from enum import Enum


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap_size", action="store", type=int, default=15)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--gap_value", action="store",  default=0)
    parser.add_argument("--dims", action="store", type=int, default=4)
    parser.add_argument("--num_samples", action="store", type=int, default=40000)
    parser.add_argument("--dir_data", action="store",  default=".")
    args = parser.parse_args()

    data = torch.load('%s/particle_spring_40000.pt'%args.dir_data, 'wb')

    data = data[0:args.num_samples,:]
    T = int(data.shape[1]/args.dims)
    data_gap = data.reshape(args.num_samples,T,args.dims).copy()
    data_gap.resize(args.num_samples,T,args.dims+1) # add feature for missing value token
    data_gap[:,:,args.dims] = 1 # add feature for missing value token
    torch.save(data_gap, open('particle_spring_40000_no_gaps.pt', 'wb'))
    gap_starts = np.random.randint(1,T-args.gap_size-1,args.num_samples)
    for i in range(args.num_samples):
        indx = gap_starts[i]
        data_gap[i,indx:indx+args.gap_size,:] = args.gap_value # make a gap
        data_gap[i,indx:indx+args.gap_size,args.dims] = 0 # set gap feature to 0 
    torch.save(data_gap, open('particle_spring_40000_gaps.pt', 'wb'))

    if args.plot:
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Input data with gaps', fontsize=30) 
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.axis([0, T,-30.0,30.0])
        def draw(i,color):
            plt.scatter(np.repeat(np.arange(T),args.dims+1), data_chunk[i,:], c=color)
            plt.savefig('gaps_%d.pdf'%i)
            plt.clf()
        for j in range(0,5):
            print("Predicting sample %d:"%j)
            draw(j,'m')

