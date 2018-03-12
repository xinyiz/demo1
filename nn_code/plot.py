import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
L = 240 # length of each input sample 
future = L 
end_idx = int(L/2)
feature_dims = 4;
def plot_loss_curve(prefix=""):
    loss = np.load("../output/%s_no_mdn_training_loss.npy"%prefix)
    x = np.linspace(0, len(loss), len(loss))
    fig = plt.figure()
    plt.title('Loss', fontsize=30) 
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.plot(x, loss);
    plt.savefig('../output/%s_training_loss.pdf'%prefix)
    plt.close()
def plot_learning_curve(prefix=""):
    x = np.load("../output/%s_loss_x.npy"%prefix)
    train_loss = np.load("../output/%s_loss_train.npy"%prefix)
    test_loss = np.load("../output/%s_loss_test.npy"%prefix)
    print(x)
    print(train_loss)

    fig = plt.figure()
    plt.title('Train Test Curve', fontsize=30) 
    plt.xlabel('num samples', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.plot(x, train_loss, 'r');
    plt.plot(x, test_loss, 'g');
    plt.savefig('../output/%s_learning_curve.pdf'%prefix)
    plt.close()
def draw(y,pred_test_input,j,prefix,color):
    yi = y[j]
    yi_all = yi.reshape(yi.shape[0]*yi.shape[1])
    plt.scatter(np.repeat(np.arange(end_idx),feature_dims), pred_test_input[j], c=color)
    s_idx = end_idx*feature_dims
    plt.scatter(np.repeat(np.arange(end_idx,end_idx + future),feature_dims), yi_all[s_idx:], c='b', marker='*')
    plt.savefig('../output/%s_spring_lstm_%d.pdf'%(prefix,j))
    plt.clf()
def plot_pred(prefix="",num_samples=3):
    y = np.load("../output/%s_prediction_future.npy"%prefix)
    pred_test_input = np.load("../output/%s_prediction_test.npy"%prefix)
    # draw the result
    plt.figure(figsize=(30,10))
    plt.title('Predict future values for time sequences\n(Values are sampled from the predicted PDF)', fontsize=30) 
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.axis([0, future+end_idx,-30.0,30.0])
    for j in range(0,num_samples):
        print("Predicting sample %d:"%j)
        draw(y,pred_test_input,j,prefix,'m')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", action="store", dest="prefix")
    parser.add_argument("--num_pred_samples", action="store", dest="num_pred_samples", type=int)
    args = parser.parse_args()
    plot_learning_curve(prefix=args.prefix)
    plot_loss_curve(prefix=args.prefix)
    plot_pred(prefix=args.prefix,num_samples=args.num_pred_samples)
