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
class BRNN(nn.Module):
    def __init__(self,n_in_features,n_out_features,hidden_dims,length,cuda=False):
        super(BRNN,self).__init__()
        self.hidden_dims = hidden_dims
        self.input_dims = n_in_features
        self.output_dims = n_out_features
        self.length = length 
        self.is_cuda=cuda

        self.lstm1 = nn.LSTM(self.input_dims,hidden_dims,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(self.hidden_dims*self.length, self.output_dims*self.length) 

    # forward pass over all time samples
    def forward(self,data):
        if self.is_cuda:
            h0 = Variable(torch.zeros(2,data.size(0),self.hidden_dims).double().cuda()) 
            c0 = Variable(torch.zeros(2,data.size(0),self.hidden_dims).double().cuda())
        else:
            h0 = Variable(torch.zeros(2,data.size(0),self.hidden_dims).double()) 
            c0 = Variable(torch.zeros(2,data.size(0),self.hidden_dims).double())
        encoding, _ = self.lstm1(data, (h0, c0)) 
        forward,backward = torch.chunk(encoding,2,2) # get values from forward and backward nets.
        merged = (forward + backward).view(data.size(0),data.size(1)*self.hidden_dims)
        out = self.fc(merged)  
        return out
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
        train_b = train[ss*(incr-1):ss*incr,:]
        train_target_b = train_target[ss*(incr-1):ss*incr,:]
        test_b = test[ss*(incr-1):ss*incr,:]
        test_target_b = test_target[ss*(incr-1):ss*incr,:]
        out_train = srnn(train_b)
        out_test = srnn(test_b)
        loss_train = criterion(out_train, train_target_b)
        loss_test = criterion(out_test, test_target_b)
        train_loss[ss] = train_loss[ss-1] + loss_train.data.cpu().numpy()[0]
        test_loss[ss] = test_loss[ss-1] + loss_test.data.cpu().numpy()[0]
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
    parser.add_argument("--dims", action="store", type=int, default=11)
    parser.add_argument("--length", action="store", type=int, default=240)
    parser.add_argument("--L", action="store",  type=int, default=21)

    parser.add_argument("--nhu", action="store",type=int,default=100)
    parser.add_argument("--split", action="store", type=float, default=0.8)
    parser.add_argument("--lr", action="store", type=float, default=0.0001)
    parser.add_argument("--num_samples", action="store", type=int, default=5000)
    parser.add_argument("--epochs", action="store", type=int, default=5)
    parser.add_argument("--bsz", action="store", type=int, default=64)
    parser.add_argument("--num_pred_samples", action="store", type=int, default=1)
    parser.add_argument("--lc_incr", action="store", type=int, default=100)
    parser.add_argument("--tag", action="store", default=str(int(time.time())))
    parser.add_argument("--dir_data", action="store",  default="/input")
    parser.add_argument("--dir_out", action="store",  default="/output")
    args = parser.parse_args()


    N = args.num_samples
    train_idx = math.floor(N*args.split)
    test_end_idx = train_idx + math.floor(N*(1.0-args.split))
    MODEL_NAME = "residual_brnn"
    out_prefix = "%s/%s_%s_%d_%d_%d"%(args.dir_out,args.tag,MODEL_NAME,args.nhu,args.bsz,args.num_samples)

    ### SETUP INPUTS AND TARGETS FOR TRAINING ###
    # input x_[t-1] should predict x_[t]
    
    data_keys = torch.load('%s/particle_spring_10000_15_keys.pt'%(args.dir_data))
    data_target = torch.load('%s/particle_spring_10000_15_target.pt'%(args.dir_data))
    data_artificial = torch.load('%s/particle_spring_50_15_interleaved.pt'%(args.dir_data))

    #### CHANGE THIS ###
    # input goes from 0 to t-1
    if(args.cuda):
        input = Variable(torch.from_numpy(data_keys[:train_idx,:]).cuda(), requires_grad=False)
        target = Variable(torch.from_numpy(data_target[:train_idx,:]).cuda(), requires_grad=False)
    else:
        input = Variable(torch.from_numpy(data_keys[:train_idx,:]), requires_grad=False)
        target = Variable(torch.from_numpy(data_target[:train_idx,:]), requires_grad=False)

    num_training_samples = input.size(0);
    T = input.size(1);
    print("Number of samples: ",num_training_samples)
    print("Length of data: ",T)
    keys_dims = input.size(2)
    target_dims = target.size(2)
    print("Num features for key frames",keys_dims)
    print("Num features for target frames",target_dims)
    
    ### BUILD MODEL ###
    srnn = BRNN(keys_dims,target_dims,args.nhu,T,args.cuda)
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
                loss = criterion(out, target_b.view(args.bsz,T*target_dims))
                loss.backward()
                train_loss[i] = loss.data.cpu().numpy()[0]
                print('loss:', train_loss[i])
                optimizer.step()
        ### SAVE MODEL ###
        torch.save(srnn.state_dict(), "%s.pth"%(out_prefix))
        np.save("%s_training_loss"%(out_prefix), train_loss)
        if(args.floyd):
            plot_loss(train_loss,out_prefix,False)
        else:
            plot_loss(train_loss,out_prefix,True)
        print("tag ",args.tag)
        
    i = args.epochs
    srnn.load_state_dict(torch.load("%s.pth"%(out_prefix)))

    ### PREDICT ###

    ### PLOT LEARNING CURVE ###
    if(args.cuda):
        test_input = Variable(torch.from_numpy(data_keys[train_idx:test_end_idx,:]).cuda(), requires_grad=False)
        test_target = Variable(torch.from_numpy(data_target[train_idx:test_end_idx,:]).cuda(), requires_grad=False)
        test_artificial = Variable(torch.from_numpy(data_artificial).cuda(), requires_grad=False)
    else:
        test_input = Variable(torch.from_numpy(data_keys[train_idx:test_end_idx,:]), requires_grad=False)
        test_target = Variable(torch.from_numpy(data_target[train_idx:test_end_idx,:]), requires_grad=False)
        test_artificial = Variable(torch.from_numpy(data_artificial), requires_grad=False)
    if(args.floyd):
        plot_learning_curve(input,target,test_input,test_target,srnn,criterion,args.lc_incr,out_prefix,False)
    else:
        plot_learning_curve(input,target,test_input,test_target,srnn,criterion,args.lc_incr,out_prefix,True)
    
    ### PLOT PREDICTION FOR TEST and ARTIFICIAL DATA###
    res = srnn(test_input)
    pred = res.data.cpu().numpy()
    pred.reshape(pred.shape[0],T*target_dims)

    input_numpy = test_input.data.cpu().numpy()
    input_numpy = input_numpy.reshape(input_numpy.shape[0],T*keys_dims)
    target_numpy = test_target.data.cpu().numpy()
    target_numpy = target_numpy.reshape(target_numpy.shape[0],T*target_dims)

    res_artificial = srnn(test_artificial)
    pred_artificial = res_artificial.data.cpu().numpy()
    pred_artificial.reshape(pred_artificial.shape[0],T*target_dims)

    input_artificial = test_artificial.data.cpu().numpy()
    input_artificial = input_artificial.reshape(input_artificial.shape[0],T*keys_dims)
    np.save("%s_pred"%out_prefix, pred)
    np.save("%s_artificial_pred"%out_prefix, pred_artificial)
    np.save("%s_artificial_input"%out_prefix, input_artificial)
    np.save("%s_target"%out_prefix, target_numpy)

    if(not args.floyd):
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Values are sampled from the predicted PDF)', fontsize=30) 
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        axes = plt.gca()
        axes.set_xlim([0,T])
        def draw(inp,pred,j,colors,features,target=np.array([])):
            for f,co in zip(features,colors):
                plt.plot(np.arange(T), inp[j,f:T*target_dims:target_dims],'+', c=co,ms=10)
                plt.plot(np.arange(T), pred[j,f:T*target_dims:target_dims], 'x',c=co,ms=3)
                if target.size != 0:
                    plt.plot(np.arange(T), target[j,f:T*target_dims:target_dims], 'o',
                            markerfacecolor='None',markeredgecolor=co,ms=5)
            if target.size!=0:
                plt.savefig('%s_gap_%d_%s.pdf'%(out_prefix,j,''.join(str(f) for f in features)))
            else:
                plt.savefig('%s_generalization_%d_%s.pdf'%(out_prefix,j,''.join(str(f) for f in features)))
            plt.clf()
        for j in range(0,args.num_pred_samples):
            print("Predicting sample %d:"%j)
            draw(input_numpy,pred,j,['g','b'],[0,1],target=target_numpy)
            draw(input_artificial,pred_artificial,j,['g','b'],[0,1])
            plt.close()
