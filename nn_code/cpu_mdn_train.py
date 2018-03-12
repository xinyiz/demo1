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
class MixtureRNN(nn.Module):
    def __init__(self,n_features,hidden_dims,num_mixtures):
        super(MixtureRNN,self).__init__()
        self.hidden_dims = hidden_dims
        self.n_features = n_features
        #self.input_dims = n_features + hidden_dims
        self.input_dims = n_features
        self.num_mixtures = num_mixtures
        self.output_dims = (n_features+2)*num_mixtures

        self.lstm1 = nn.LSTMCell(self.input_dims,hidden_dims)
        self.lstm2 = nn.LSTMCell(hidden_dims,self.output_dims)
        #self.i2h = nn.Linear(self.input_dims,hidden_dims)
        #self.h2o = nn.Linear(hidden_dims,self.output_dims)

    @staticmethod
    def get_mixture_coeffs(nn_output,num_mixtures,n_features):
        """
        Computes the mixture model parameters from the network output

        param nn_output: the output from the network

        return: out_pi containing mixture coefficients of size (batch_size,num_mixtures)
                    e.g
                    input 1 [[ pi1,pi2,pi3,...],
                    input 2  [ pi1,pi2,pi3,...],
                    .        [................],
                    input n  [ pi1,pi2,pi3,...]]
                out_sigma containing variances of size (input_size,num_mixtures)
                    e.g
                    input 1 [[ sig1,sig2,sig3,...],
                    input 2  [ sig1,sig2,sig3,...],
                    .        [...................],
                    input n  [ sig1,sig2,sig3,...]]
                out_mu containing mixture coefficients of size (input_size, num_mixtures*num_features)
                    e.g
                    input 1 [[ mix1_mu1,mix1_mu2,mix1_mu3 | mix2_mu1, mix2_mu2, mix2_mu3 |.... ],
                    input 2  [ mix1_mu1,mix1_mu2,mix1_mu3 | mix2_mu1, mix2_mu2, mix2_mu3 |.... ],
                    input 3  [ mix1_mu1,mix1_mu2,mix1_mu3 | mix2_mu1, mix2_mu2, mix2_mu3 |.... ],
                    .        [................................................................ ],
                    input n  [ mix1_mu1,mix1_mu2,mix1_mu3 | mix2_mu1, mix2_mu2, mix2_mu3 |.... ]]
        """
        out_pi = nn_output[:,0:num_mixtures] # num_mixtures
        out_sigma = nn_output[:,num_mixtures:2*num_mixtures] # num_mixtures
        out_mu = nn_output[:,2*num_mixtures:] # num_mixtures * n_features
        # Compute Sigma
        out_sigma = torch.exp(out_sigma)
        # Compute PI
        sm = torch.nn.Softmax()
        out_pi = sm(out_pi)
        return out_pi, out_sigma, out_mu

    @staticmethod
    def loss_func(output,target,num_mixtures,L,n_out_features):
        output_at_t = torch.chunk(output,L,1)
        target_at_t = torch.chunk(target,L,1)
        total_loss = 0
        for output_t,target_t in zip(output_at_t,target_at_t):
            output_t = output_t.squeeze(1)
            #print("output_t 0", output_t.size(0))
            #print("output_t 1", output_t.size(1)) (k+2)*L
            out_pi,out_sigma,out_mu = MixtureRNN.get_mixture_coeffs(output_t,num_mixtures,n_out_features)
            for j in range(num_mixtures):
                # Gaussian
                num_term = torch.norm((target_t-out_mu[:,j*n_out_features:(j+1)*n_out_features]),2,1)
                denom_term = 2.0*torch.pow(out_sigma[:,j],2.0)            
                gaussian_exp = torch.exp(-torch.div(num_term,denom_term))
                gaussian_coeff = 1.0/(math.pow(2*math.pi,n_out_features/2.0)*torch.pow(out_sigma[:,j],n_out_features))
                gaussian = torch.mul(gaussian_exp,gaussian_coeff)
                # Combine rest of terms (likelihood)
                result = torch.mul(out_pi[:,j],gaussian)
                result = -torch.log(result)
                total_loss = total_loss + result
        total_loss = torch.mean(total_loss)    
        return total_loss

    def compute_means(self,pis,mus):
        """
        return: array of size (batch_size,n_features) containing means
        of each feature
            e.g
            input 1 [[f1_mean,f2_mean,f3_mean,..],
            input 2  [f1_mean,f2_mean,f3_mean,..],
            .        [..........................],
            input n  [f1_mean,f2_mean,f3_mean,..]]
        """
        result = np.zeros((pis.shape[0],self.n_features))
        for i,mix_i_mu in enumerate(np.split(mus,self.num_mixtures,1)):
            a = pis[:,i]
            b = mix_i_mu
            mean = np.multiply(a[:,np.newaxis],mix_i_mu)
            result = result + mean
        return Variable(torch.from_numpy(result))

    # forward pass over all time samples
    def forward(self,data,future = 0,num_samples=1):
        outputs = []
        # data.size(0) is the number of input example
        # data.size(1) is the number of time samples for each input example

        h_t = Variable(torch.zeros(data.size(0), self.hidden_dims).double(), requires_grad=False)
        c_t = Variable(torch.zeros(data.size(0), self.hidden_dims).double(), requires_grad=False)
        h_t2 = Variable(torch.zeros(data.size(0), self.output_dims).double(), requires_grad=False)
        output = Variable(torch.zeros(data.size(0), self.output_dims).double(), requires_grad=False)

        #hidden = Variable(torch.zeros(data.size(0), self.hidden_dims).double(), requires_grad=False)
        time_samples = int(data.size(1)/self.n_features)
        for i, input_t in enumerate(data.chunk(time_samples, dim=1)):
            # input_t.size(0) is the number of input examples
            # input_t.size(1) is the feature dimension of each time sample

            # RNN takes in the previous hidden state and the current input
            # Each row of input into nn.Linear is assumed to be an input of a given batch
            #inputs = torch.cat((input_t, hidden), 1) # concat along feature dimension

            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, output = self.lstm2(c_t, (h_t2, output))
            #hidden = self.i2h(inputs)
            #output = self.h2o(hidden)
            outputs += [output]
        if(future != 0):
            y = []
            # First sample from the training data
            for output_t in outputs:
                out_pi,out_sigma,out_mu = MixtureRNN.get_mixture_coeffs(output_t,self.num_mixtures,self.n_features)
                pis = out_pi.data.numpy() 
                sigmas = out_sigma.data.numpy() 
                mus = out_mu.data.numpy() 
                y_sample = sample_gaussian_mixture(pis,sigmas,mus,num_samples,self.n_features)
                y += [y_sample]
            t_prev = self.compute_means(pis,mus)
            # Now predict 
            for i in range(future):# if we should predict the future
                #inputs = torch.cat((t_prev, hidden), 1) # concat along feature dimension
                
                h_t, c_t = self.lstm1(t_prev, (h_t, c_t))
                h_t2, output_t = self.lstm2(c_t, (h_t2, output))

                #hidden = self.i2h(inputs)
                #output_t = self.h2o(hidden)
                out_pi,out_sigma,out_mu = MixtureRNN.get_mixture_coeffs(output_t,self.num_mixtures,self.n_features)
                pis = out_pi.data.numpy() 
                sigmas = out_sigma.data.numpy() 
                mus = out_mu.data.numpy() 
                y_sample = sample_gaussian_mixture(pis,sigmas,mus,num_samples,self.n_features)
                t_prev = y_sample[:,0:self.n_features]
                y += [y_sample]
            y = torch.stack(y,1) 
            return y
        # individual outputs at t to get final predicted sequence
        outputs = torch.stack(outputs,1).squeeze(2) #get rid of extraneous last dim with squeeze 
        return outputs

def sample_gaussian_mixture(pis,sigmas,mus,num_samples,n_features):
    """
    return: array of size (batch_size,num_samples*n_features) containing samples 
    taken from the gaussian mixture parameratized by pis, sigmas, mus
        e.g
        input 1 [[ s1_f1,s1_f2,s1_f3 | s2_f1, s2_f2, s2_f3 |.... ],
        input 2  [ s1_f1,s1_f2,s1_f3 | s2_f1, s2_f2, s2_f3 |.... ],
        input 3  [ s1_f1,s1_f2,s1_f3 | s2_f1, s2_f2, s2_f3 |.... ],
        .        [...............................................],
        input n  [ s1_f1,s1_f2,s1_f3 | s2_f1, s2_f2, s2_f3 |.... ]]
    """
    # Gaussian PDF parameters
    batch_size = pis.shape[0]
    num_mixtures = pis.shape[1]
    samples = np.zeros((batch_size,num_samples*n_features))
    gmm = GaussianMixture(n_components=num_mixtures, covariance_type='spherical')
    gmm.fit(np.random.rand(10,1))  # Now it thinks it is trained
    for i in range(batch_size):
        gmm.weights_ = pis[i]
        gmm.means_ = mus[i].reshape(num_mixtures,n_features)
        gmm.covariances_ = np.expand_dims(sigmas[i],axis=1)**2
        sample = gmm.sample(num_samples)
        samples[i] = np.ravel(sample[0])
    return Variable(torch.from_numpy(samples))

def batch_data(in_data,target_data,batch_size):
    return [torch.split(in_data,batch_size,0),torch.split(target_data,batch_size,0)]

def plot_loss(loss):
    print("here")
    x = np.linspace(0, len(loss), len(loss))
    fig = plt.figure()
    plt.title('Loss)', fontsize=30) 
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.plot(x, loss);
    plt.savefig('training_loss.pdf')
    plt.close()
if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    L = 240 # length of each input sample 
    N = 5000 # number of input samples(sine waves)
    number_hu = 50 
    num_mixtures = 1 
    train_test_split = 0.995 # proportion of samples to use for training/prediction   
    train_idx = math.floor(N*train_test_split)
    EPOCHS = 10 
    BATCH_SIZE = 64 
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
    srnn = MixtureRNN(feature_dims,number_hu,num_mixtures)
    srnn.double() # set model parameters to double

    train_loss = np.zeros(EPOCHS)
    if(not PREDICT):
        ### DEFINE LOSS ###
        criterion = MixtureRNN.loss_func
        optimizer = optim.Adam(srnn.parameters(),lr=0.00001)

        ### TRAIN MODEL ###
        for i in range(EPOCHS):
            print('EPOCH: ', i)
            in_batches,target_batches = batch_data(input,target,BATCH_SIZE)
            for input_b,target_b in zip(in_batches,target_batches):
                out = srnn(input_b)
                optimizer.zero_grad()
                loss = criterion(out, target_b,num_mixtures,L,feature_dims)
                loss.backward()
                train_loss[i] = loss.data.numpy()[0]
                print('loss:', train_loss[i])
                optimizer.step()
        ### SAVE MODEL ###
        torch.save(srnn.state_dict(), "/output/particle_spring.pth")
        np.save("/output/training_loss", train_loss)
        plot_loss(train_loss)
    ## TODO 

    ###PREDICT###
    else:
        srnn.load_state_dict(torch.load("input/particle_spring.pth"))
        i = EPOCHS
        future = L 
        end_idx = int(L/2)
        test_input = Variable(torch.from_numpy(data[train_idx: , :end_idx*feature_dims]), requires_grad=False)
        num_pred_samples = 1
        pred = srnn(test_input, future = future, num_samples=num_pred_samples)
        print("PREDICTION:")
        print(pred.size(0))
        print(pred.size(1))

        y = pred.data.numpy()
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
            yi_all = yi.reshape(yi.shape[0]*yi.shape[1]) # size (T,n_samples*n_features) -> size T*n*samples_n_features
            print("test_input",test_input[j])
            plt.scatter(np.repeat(np.arange(end_idx),feature_dims), test_input[j].data.numpy(), c=color)
            s_idx = end_idx*num_pred_samples*feature_dims
            plt.scatter(np.repeat(np.arange(end_idx,end_idx + future),num_pred_samples*feature_dims), yi_all[s_idx:], c='b', marker='*')
            plt.savefig('spring_lstm%d.pdf'%j)
            plt.clf()
        for j in range(0,1):
            print("sample %d prediction:"%j)
            draw(y,j, 'm')
        plt.close()

