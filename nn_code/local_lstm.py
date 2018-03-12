from __future__ import print_function
import math
import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import time
import argparse
class LSTMRNN(nn.Module):
    def __init__(self,n_features,hidden_dims):
        super(LSTMRNN,self).__init__()
        self.hidden_dims = hidden_dims
        self.n_features = n_features
        self.input_dims = n_features
        self.output_dims = n_features

        self.lstm1 = nn.LSTMCell(self.input_dims,hidden_dims)
        self.lstm2 = nn.LSTMCell(hidden_dims,self.output_dims)

    # forward pass over all time samples
    def forward(self,data,future = 0):
        outputs = []

        h_t = Variable(torch.zeros(data.size(0), self.hidden_dims).double(), requires_grad=False)
        c_t = Variable(torch.zeros(data.size(0), self.hidden_dims).double(), requires_grad=False)
        h_t2 = Variable(torch.zeros(data.size(0), self.output_dims).double(), requires_grad=False)
        output = Variable(torch.zeros(data.size(0), self.output_dims).double(), requires_grad=False)

        #hidden = Variable(torch.zeros(data.size(0), self.hidden_dims).double(), requires_grad=False)
        time_samples = int(data.size(1)/self.n_features)
        for i, input_t in enumerate(data.chunk(time_samples, dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, output = self.lstm2(c_t, (h_t2, output))
            outputs += [output]
        for i in range(future):# if we should predict the future
            #inputs = torch.cat((t_prev, hidden), 1) # concat along feature dimension
            
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, output = self.lstm2(c_t, (h_t2, output))
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
    if cuda:
        return [in_batches.cuda(),target_batches.cuda()]
    else:
        return [in_batches,target_batches]

def plot_loss(loss,prefix="",plot=False):
    if(plot):
        x = np.linspace(0, len(loss), len(loss))
        fig = plt.figure()
        plt.title('Loss', fontsize=30) 
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('loss', fontsize=20)
        plt.plot(x, loss);
        plt.savefig('../output/%s_training_loss.pdf'%prefix)
        plt.close()
    else:
        return

def plot_learning_curve(train,train_target,test,test_target,model,criterion,incr,prefix="",plot=False):
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
        train_loss[ss] = loss_train.data.numpy()[0]
        test_loss[ss] = loss_test.data.numpy()[0]
    x = np.linspace(incr, num_samples*incr, num_samples)
    if(plot):
        fig = plt.figure()
        plt.title('Train Test Curve', fontsize=30) 
        plt.xlabel('num samples', fontsize=20)
        plt.ylabel('loss', fontsize=20)
        plt.plot(x[:-1], train_loss[1:], 'r');
        plt.plot(x[:-1], test_loss[1:], 'g');
        plt.savefig('../output/%s_learning_curve.pdf'%prefix)
        plt.close('all')
    else:
        np.save("../output/%s_loss_x"%PREFIX, x[:-1])
        np.save("../output/%s_loss_train"%PREFIX, train_loss[1:])
        np.save("../output/%s_loss_test"%PREFIX, test_loss[1:])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", help="increase output verbosity",action="store_true")
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    L = 240 # length of each input sample 
    N = 5000 # number of input samples(sine waves)
    number_hu = 10 
    train_test_split = 0.9 # proportion of samples to use for training/prediction   
    train_idx = math.floor(N*train_test_split)
    test_end_idx = train_idx + math.floor(N*(1.0-train_test_split))
    EPOCHS = 40 
    BATCH_SIZE = 128 
    NUM_PRED_SAMPLES = 2 
    LC_INCR = 100 
    PREFIX = 1495049263 
    PREDICT = True 

    ### SETUP INPUTS AND TARGETS FOR TRAINING ###
    # input x_[t-1] should predict x_[t]
    
    data = torch.load('../sim_data/particle_spring_10000.pt')

    ### CHANGE THIS ###
    feature_dims = 4;
    # input goes from 0 to t-1
    input = Variable(torch.from_numpy(data[:train_idx, :-1*feature_dims]), requires_grad=False)
    num_training_samples = input.size(0);
    print("num_training_samples is",num_training_samples)
    num_time_samples = input.size(1);
    
    # target goes from 1 to t 
    target = Variable(torch.from_numpy(data[:train_idx, 1*feature_dims:]), requires_grad=False)
    ### BUILD MODEL ###
    srnn = LSTMRNN(feature_dims,number_hu)
    srnn.double() # set model parameters to double
    if(args.cuda):
        srnn.cuda()
    criterion = nn.MSELoss()
    if (not PREDICT):

        train_loss = np.zeros(EPOCHS)

        ### DEFINE LOSS ###
        optimizer = optim.Adam(srnn.parameters(),lr=0.0001)

        ### TRAIN MODEL ###
        for i in range(EPOCHS):
            print('EPOCH: ', i)
            in_batches,target_batches = batch_data(input,target,BATCH_SIZE,args.cuda)
            for input_b,target_b in zip(in_batches,target_batches):
                out = srnn(input_b)
                optimizer.zero_grad()
                loss = criterion(out, target_b)
                loss.backward()
                train_loss[i] = loss.data.numpy()[0]
                print('loss:', train_loss[i])
                optimizer.step()
        ### SAVE MODEL ###
        torch.save(srnn.state_dict(), "../output/%s_no_mdn_particle_spring.pth"%PREFIX)
        np.save("../output/%s_no_mdn_training_loss"%PREFIX, train_loss)
        plot_loss(train_loss,PREFIX,True)
    else:
        ### PREDICT ###
        srnn.load_state_dict(torch.load("../output/%s_no_mdn_particle_spring.pth"%PREFIX))
        i = EPOCHS
        future = L 
        end_idx = int(L/2)

        ### PLOT LEARNING CURVE ###
        test_input = Variable(torch.from_numpy(data[train_idx:test_end_idx , :-1*feature_dims]), requires_grad=False)
        test_target = Variable(torch.from_numpy(data[train_idx:test_end_idx , 1*feature_dims:]), requires_grad=False)
        plot_learning_curve(input,target,test_input,test_target,srnn,criterion,LC_INCR,PREFIX,True)
        
        ### PLOT PREDICTION ###
        pred_test_input = Variable(torch.from_numpy(data[train_idx:test_end_idx , :end_idx*feature_dims]), requires_grad=False)
        if(args.cuda):
            pred_test_input = pred_test_input.cuda()
        pred = srnn(pred_test_input, future = future)

        y = pred.data.numpy()
        pred_numpy = pred_test_input.data.numpy()
        np.save("../output/%s_prediction_future"%PREFIX, y)
        np.save("../output/%s_prediction_test"%PREFIX, pred_numpy)

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
            plt.scatter(np.repeat(np.arange(end_idx),feature_dims), pred_test_input[j].data.numpy(), c=color)
            s_idx = end_idx*feature_dims
            print("yi size", yi_all.shape)
            print("index", np.repeat(np.arange(end_idx,end_idx + future),feature_dims).shape)
            plt.scatter(np.repeat(np.arange(end_idx,end_idx + future),feature_dims), yi_all[s_idx:], c='b', marker='*')
            plt.savefig('../output/%s_spring_lstm_%d.pdf'%(PREFIX,j))
            plt.clf()
        for j in range(0,NUM_PRED_SAMPLES):
            print("Predicting sample %d:"%j)
            draw(y,j, 'm')
    plt.close()
