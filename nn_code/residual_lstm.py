from __future__ import print_function
import math
import torch
import torch.nn as nn 
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import time
import argparse
class RESIDUAL_LSTM(nn.Module):
    def __init__(self,n_features,hidden_dims,cuda=False):
        super(RESIDUAL_LSTM,self).__init__()
        self.hidden_dims = hidden_dims
        self.n_features = n_features
        self.input_dims = n_features
        self.output_dims = n_features
        self.is_cuda = cuda

        self.lstm1 = nn.LSTMCell(self.input_dims,hidden_dims)
        self.lstm2 = nn.LSTMCell(hidden_dims,self.output_dims)

    # forward pass over all time samples
    def forward(self,data,future = 0):
        outputs = []
        if self.is_cuda:
            h_t = Variable(torch.zeros(data.size(0), self.hidden_dims).double().cuda(), requires_grad=False)
            c_t = Variable(torch.zeros(data.size(0), self.hidden_dims).double().cuda(), requires_grad=False)
            h_t2 = Variable(torch.zeros(data.size(0), self.output_dims).double().cuda(), requires_grad=False)
            output = Variable(torch.zeros(data.size(0), self.output_dims).double().cuda(), requires_grad=False)
            delta = Variable(torch.zeros(data.size(0), self.output_dims).double().cuda(), requires_grad=False)
        else:
            h_t = Variable(torch.zeros(data.size(0), self.hidden_dims).double(), requires_grad=False)
            c_t = Variable(torch.zeros(data.size(0), self.hidden_dims).double(), requires_grad=False)
            h_t2 = Variable(torch.zeros(data.size(0), self.output_dims).double(), requires_grad=False)
            output = Variable(torch.zeros(data.size(0), self.output_dims).double(), requires_grad=False)
            delta = Variable(torch.zeros(data.size(0), self.output_dims).double(), requires_grad=False)

        #hidden = Variable(torch.zeros(data.size(0), self.hidden_dims).double(), requires_grad=False)
        time_samples = int(data.size(1)/self.n_features)
        for i, input_t in enumerate(data.chunk(time_samples, dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, delta = self.lstm2(c_t, (h_t2, delta))
            output = delta + input_t  # we want to predict deltas
            outputs += [output]
        for i in range(future):# if we should predict the future
            #inputs = torch.cat((t_prev, hidden), 1) # concat along feature dimension
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, delta = self.lstm2(c_t, (h_t2, delta))
            output = output + delta # we want to predict deltas
            outputs += [output]
        # individual outputs at t to get final predicted sequence
        outputs = torch.stack(outputs,1).squeeze(2) #get rid of extraneous last dim with squeeze 
        return outputs
def batch_data(in_data,target_data,batch_size,cuda=False):
    nbatch = in_data.size(0) // batch_size
    in_data = in_data.narrow(0, 0, nbatch * batch_size)
    target_data = target_data.narrow(0, 0, nbatch * batch_size)
    in_batches = torch.split(in_data,batch_size,0)
    target_batches = torch.split(target_data,batch_size,0)
    return [in_batches,target_batches]

def plot_loss(loss,out_prefix="",plot=False):
    if(plot):
        x = np.linspace(0, len(loss), len(loss))
        fig = plt.figure()
        plt.title('Loss', fontsize=30) 
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('loss', fontsize=20)
        plt.plot(x, loss);
        plt.savefig('%s_training_loss.pdf'%(out_prefix))
        plt.close()
    else:
        return

def plot_learning_curve(train,train_target,test,test_target,model,criterion,incr,out_prefix="",plot=False):
    num_samples = math.floor(test.size(0)/incr)
    train_loss = np.zeros(num_samples)
    test_loss = np.zeros(num_samples)
    for ss in range(1,num_samples):
        print("Getting training and test loss with %d samples:"%(ss*incr))
        train_b = train[:ss*incr,:]
        train_target_b = train[:ss*incr,:]
        test_b = test[:ss*incr,:]
        test_target_b = test_target[:ss*incr,:]
        out_train = srnn(train_b)
        out_test = srnn(test_b)
        loss_train = criterion(out_train, train_target_b)
        loss_test = criterion(out_test, test_target_b)
        train_loss[ss] = loss_train.data.cpu().numpy()[0]
        test_loss[ss] = loss_test.data.cpu().numpy()[0]
    x = np.linspace(incr, num_samples*incr, num_samples)
    if(plot):
        fig = plt.figure()
        plt.title('Train Test Curve', fontsize=30) 
        plt.xlabel('num samples', fontsize=20)
        plt.ylabel('loss', fontsize=20)
        plt.plot(x[:-1], train_loss[1:], 'r');
        plt.plot(x[:-1], test_loss[1:], 'g');
        plt.savefig('%s_learning_curve.pdf'%(out_prefix))
        plt.close('all')
    else:
        np.save("%s_loss_x"%(out_prefix), x[:-1])
        np.save("%s_loss_train"%(out_prefix), train_loss[1:])
        np.save("%s_loss_test"%(out_prefix), test_loss[1:])

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--floyd", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--dims", action="store", type=int, default=4)
    parser.add_argument("--length", action="store", type=int, default=240)

    parser.add_argument("--nhu", action="store",type=int,default=100)
    parser.add_argument("--split", action="store", type=float, default=0.8)
    parser.add_argument("--lr", action="store", type=float, default=0.0001)
    parser.add_argument("--num_samples", action="store", type=int, default=5000)
    parser.add_argument("--epochs", action="store", type=int, default=5)
    parser.add_argument("--bsz", action="store", type=int, default=64)
    parser.add_argument("--num_pred_samples", action="store", type=int, default=1)
    parser.add_argument("--lc_incr", action="store", type=int, default=100)
    parser.add_argument("--tag", action="store", default=str(int(time.time())))
    parser.add_argument("--dir_data", action="store",  default="/input/sim_data")
    parser.add_argument("--dir_out", action="store",  default="/output")
    args = parser.parse_args()


    N = args.num_samples
    L = 240 # length of each input sample 
    train_idx = math.floor(N*args.split)
    test_end_idx = train_idx + math.floor(N*(1.0-args.split))
    MODEL_NAME = "residual_lstm"
    out_prefix = "%s/%s_%s_%d_%d_%d"%(args.dir_out,args.tag,MODEL_NAME,args.nhu,args.bsz,args.num_samples)

    ### SETUP INPUTS AND TARGETS FOR TRAINING ###
    # input x_[t-1] should predict x_[t]
    
    data = torch.load('%s/particle_spring_10000.pt'%args.dir_data)

    #### CHANGE THIS ###
    # input goes from 0 to t-1
    if(args.cuda):
        input = Variable(torch.from_numpy(data[:train_idx, :-1*args.dims]).cuda(), requires_grad=False)
        target = Variable(torch.from_numpy(data[:train_idx, 1*args.dims:]).cuda(), requires_grad=False)
    else:
        input = Variable(torch.from_numpy(data[:train_idx, :-1*args.dims]), requires_grad=False)
        target = Variable(torch.from_numpy(data[:train_idx, 1*args.dims:]), requires_grad=False)
        
    num_training_samples = input.size(0);
    print("num_training_samples is",num_training_samples)
    num_time_samples = input.size(1);
    
    # target goes from 1 to t 
    ### BUILD MODEL ###
    srnn = RESIDUAL_LSTM(args.dims,args.nhu,args.cuda)
    srnn.double() # set model parameters to double
    if(args.cuda):
        srnn.cuda()
    criterion = nn.MSELoss()
    if (args.train or args.floyd):
        train_loss = np.zeros(args.epochs)

        ### DEFINE LOSS ###
        optimizer = optim.Adam(srnn.parameters(),lr=args.lr)

        ### TRAIN MODEL ###
        for i in range(args.epochs):
            print('EPOCH: ', i)
            in_batches,target_batches = batch_data(input,target,args.bsz,args.cuda)
            for input_b,target_b in zip(in_batches,target_batches):
                out = srnn(input_b)
                optimizer.zero_grad()
                loss = criterion(out, target_b)
                loss.backward()
                train_loss[i] = loss.data.cpu().numpy()[0]
                print('loss:', train_loss[i])
                optimizer.step()
        ### SAVE MODEL ###
        torch.save(srnn.state_dict(), "%s_particle_spring.pth"%out_prefix)
        np.save("%s_training_loss"%out_prefix, train_loss)
        if(args.floyd):
            plot_loss(train_loss,out_prefix,False)
        else:
            plot_loss(train_loss,out_prefix,True)
        print("tag ",args.tag)
        
    i = args.epochs
    future = L 
    end_idx = int(L/2)
    srnn.load_state_dict(torch.load("%s_particle_spring.pth"%out_prefix))

    if(args.floyd or not args.train):
        ### PREDICT ###

        ### PLOT LEARNING CURVE ###
        if(args.cuda):
            test_input = Variable(torch.from_numpy(data[train_idx:test_end_idx , :-1*args.dims]).cuda(), requires_grad=False)
            test_target = Variable(torch.from_numpy(data[train_idx:test_end_idx , 1*args.dims:]).cuda(), requires_grad=False)
        else:
            test_input = Variable(torch.from_numpy(data[train_idx:test_end_idx , :-1*args.dims]), requires_grad=False)
            test_target = Variable(torch.from_numpy(data[train_idx:test_end_idx , 1*args.dims:]), requires_grad=False)
            
        if(args.floyd):
            plot_learning_curve(input,target,test_input,test_target,srnn,criterion,args.lc_incr,out_prefix,False)
        else:
            plot_learning_curve(input,target,test_input,test_target,srnn,criterion,args.lc_incr,out_prefix,True)
        
        ### PLOT PREDICTION ###
        if(args.cuda):
            pred_test_input = Variable(torch.from_numpy(data[train_idx:test_end_idx , :end_idx*args.dims]).cuda(), requires_grad=False)
        else:
            pred_test_input = Variable(torch.from_numpy(data[train_idx:test_end_idx , :end_idx*args.dims]), requires_grad=False)
            
        pred = srnn(pred_test_input, future = future)

        y = pred.data.cpu().numpy()
        pred_numpy = pred_test_input.data.cpu().numpy()
        np.save("%s_prediction_future"%out_prefix, y)
        np.save("%s_prediction_test"%out_prefix, pred_numpy)

        if(not args.floyd):
            # draw the result
            plt.figure(figsize=(30,10))
            plt.title('Predict future values for time sequences\n(Values are sampled from the predicted PDF)', fontsize=30) 
            plt.xlabel('x', fontsize=20)
            plt.ylabel('y', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.axis([0, future+end_idx,-30.0,30.0])
            def draw(y,j,color):
                yi = y[j]
                yi_all = yi.reshape(yi.shape[0]*yi.shape[1])
                plt.scatter(np.repeat(np.arange(end_idx),args.dims), pred_test_input[j].data.numpy(), c=color)
                s_idx = end_idx*args.dims
                print("yi size", yi_all.shape)
                print("index", np.repeat(np.arange(end_idx,end_idx + future),args.dims).shape)
                plt.scatter(np.repeat(np.arange(end_idx,end_idx + future),args.dims), yi_all[s_idx:], c='b', marker='*')
                plt.savefig('%s_spring_%d.pdf'%(out_prefix,j))
                plt.clf()
            for j in range(0,args.num_pred_samples):
                print("Predicting sample %d:"%j)
                draw(y,j, 'm')
            plt.close()
