import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pylab as plt
from scipy.stats import ortho_group
from scipy.stats import linregress
import time
from torch.optim.lr_scheduler import MultiStepLR
from torch import optim
import logging
import argparse

def train(dataloader, model, loss_fn, optimizer):
    """
    trains one epoch
    """
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        #Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred.flatten(), y)

        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def add_weight_decay(model, weight_decay, skip_list=()):
    """
    helper function for weight decay only on weights, not on biases
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  #frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def weight_decay_val(paramlist):
    """
    computes the value of the weight decay term after an epoch
    """
    sum = 0.
    for weight in paramlist[1]['params']:
        with torch.no_grad():
            sum += (weight**2).sum()
    return sum.item()

def gen_data(filename,device,datasetsize,r,seed,trainsize=2**18,testsize=2**10,d=20,funcseed=42,verbose=False,ood=False,std=0):
    """
    data generating function
    """
    ##Generate data with a true central subspaces of varying dimensions
    #generate X values for training and test sets
    np.random.seed(seed) #set seed for data generation
    trainX = np.random.rand(d,trainsize).astype(np.float32)[:,:datasetsize] - 0.5 #distributed as U[-1/2, 1/2]
    testX = np.random.rand(d,testsize).astype(np.float32) - 0.5 #distributed as U[-1/2, 1/2]
    #out of distribution datagen
    if ood:
      trainX *= 2 #now distributed as U[-1, 1]
      testX *= 2 #now distributed as U[-1, 1]
    ##for each $r$ value create and store data-gen functions and $y$ evaluations
    #geneate params for functions
    k = d+1
    np.random.seed(funcseed) #set seed for random function generation
    U = ortho_group.rvs(k)[:,:r]
    Sigma = np.random.rand(r)*100
    V = ortho_group.rvs(d)[:,:r]
    W = (U * Sigma) @ V.T
    A = np.random.randn(k)
    B = np.random.rand(k) - 1/2
    np.save(filename+f"/r{r}U",U.copy())
    np.save(filename+f"/r{r}Sigma",Sigma.copy())
    np.save(filename+f"/r{r}V",V.copy())
    np.save(filename+f"/r{r}W",W.copy())
    np.save(filename+f"/r{r}A",A.copy())
    np.save(filename+f"/r{r}B",B.copy())
    #create functions
    def g(z): #active subspace function
        hidden_layer = (U*Sigma)@z
        hidden_layer = hidden_layer.T + B
        hidden_layer = np.maximum(0,hidden_layer).T
        return A@hidden_layer
    def f(x): #teacher network
        z = V.T@x
        logging.info(x.shape)
        eps = std*np.random.randn(x.shape[1])
        logging.info(eps.shape)
        logging.info(g(z).shape)
        return g(z) + eps
    #generate data
    trainY = f(trainX).astype(np.float32)
    testY = f(testX).astype(np.float32)
    #move data to device
    trainX = torch.from_numpy(trainX).T.to(device)
    trainY = torch.from_numpy(trainY).to(device)
    testX = torch.from_numpy(testX).T.to(device)
    testY = torch.from_numpy(testY).to(device)
    if verbose:
        logging.info("trainX shape = {} trainY shape = {}".format(
            trainX.shape,
            trainY.shape
        ))
        logging.info("first entry of trainY {}".format(
            trainY[0]
        ))
    return trainX,trainY,testX,testY

def Llayers(L,d,width):
    """
    model class. Construct L-1 linear layers; bias terms only on last linear layer and final relu layer.
    """
    if L < 2:
        raise ValueError("L must be at least 2")
    if L == 2:
        linear_layers = [nn.Linear(d,width,bias=True)]
    if L > 2:
        linear_layers = [nn.Linear(d,width,bias=False)]
        for l in range(L-3):
            linear_layers.append(nn.Linear(width,width,bias=False))
        linear_layers.append(nn.Linear(width,width,bias=True))

    relu = nn.ReLU()

    last_layer = nn.Linear(width,1)

    layers = linear_layers + [relu,last_layer]

    return nn.Sequential(*layers)

def train_L_layers(filename,datasetsize,L,r,weight_decay,epochs=30_100,lr=1e-4,
                   trainsize=2**18,testsize=2**10,d=20,funcseed=42,datagenseed=1,
                   initseed=42,batch_size=64,width=1000,verbose=False,
                   no_wd_last_how_many_epochs=100,std=0,
                   scheduler=None,**schedulerkwargs):
    starttime = time.time()
    paramname = f"N{datasetsize}_L{L}_r{r}_wd{weight_decay}_epochs{epochs}"

    # check GPU is enabled
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"{paramname}: device is {device}")

    #generate data
    logging.info(f"{paramname}: generating data")
    trainX,trainY,testX,testY = gen_data(filename,device,datasetsize,r,datagenseed,trainsize,testsize,d,funcseed,verbose,std=std)

    #define pytorch dataloaders
    dataset = torch.utils.data.TensorDataset(trainX,trainY) #create your dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) #create your dataloader

    #initialize model
    torch.manual_seed(initseed) #set seed for initalization
    model = Llayers(L,d,width)
    model.to(device)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])

    loss_fn = nn.MSELoss()
    paramlist = add_weight_decay(model,weight_decay)
    optimizer = torch.optim.Adam(paramlist, lr=lr)

    if verbose:
        logging.info(f"{paramname}: lambda = {paramlist[1]['weight_decay']}")

    #main training loop
    with torch.no_grad():
        trainmse = torch.zeros(epochs,device=device)
        weightdecay = torch.zeros(epochs,device=device)
        learningrate = torch.zeros(epochs,device=device)
    if verbose:
        printfreq = 1000 if datasetsize > 500 else 5000
    if verbose:
        logging.info("Time: {:.1f} Starting to Train".format(
            time.time()-starttime
        ))
    if scheduler is not None:
        scheduler = scheduler(optimizer,**schedulerkwargs)
    flag = False
    for t in range(epochs):
        train(dataloader, model, loss_fn, optimizer)

        with torch.no_grad():
            #record current MSE, weight decay value, and learning rate
            trainmse[t] = loss_fn(model(trainX).flatten(),trainY)
            for param in model.parameters():
                weightdecay[t] += weight_decay_val(paramlist)
            if scheduler is not None:
                learningrate[t] = scheduler.optimizer.param_groups[0]['lr']

            #report loss every few epochs
            if verbose:
                if t % printfreq == 0:
                    logging.info("Time: {:.1f} Epoch: {} Train MSE:{:.6e} Weight Decay:{:.6e}".format(
                        time.time()-starttime,
                        t,
                        trainmse[t],
                        weightdecay[t]
                    ))

        #adjust learning rate
        if scheduler is not None:
            scheduler.step()

        #turn off weight decay for last 100 epochs
        if flag is False and t > epochs - no_wd_last_how_many_epochs:
            optimizer.param_groups[1]['weight_decay'] = 0
            flag = True

    #report loss at end of training
    with torch.no_grad():
        testmse = loss_fn(model(testX).flatten(),testY).detach().cpu().numpy()
        if verbose:
            logging.info("Time: {:.1f} Test MSE:{:.6e} Train MSE:{:.6e} Weight Decay:{:.6e}".format(
                time.time()-starttime,
                testmse,
                trainmse[t],
                weightdecay[t]
            ))

    if verbose:
        logging.info(f"{paramname}: trained in {time.time()-starttime} seconds")
    return model,trainmse,weightdecay,learningrate,testmse

if __name__ == "__main__":
    #parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, help = "name of file")
    parser.add_argument("--datasetsize", type=int, help = "number of samples to train with")
    parser.add_argument("--L", type=int, help = "number of layers")
    parser.add_argument("--r", type=int, help = "rank of ground truth function")
    parser.add_argument("--labelnoise", type=float, help = "standard deviation of normally distributed label noise")
    parser.add_argument("--weight_decay", type=float, help = "regularization parameter")
    parser.add_argument("--epochs", type=int, help = "number of epochs to train for")

    args = parser.parse_args()

    #set args
    datasetsize = args.datasetsize
    L = args.L
    r = args.r
    std = args.labelnoise
    weight_decay = args.weight_decay
    epochs = args.epochs
    filename = args.filename
    paramname = f"N{datasetsize}_L{L}_r{r}_wd{weight_decay}_epochs{epochs}"
    
    #set up logging
    logging.basicConfig(filename=f"log/{filename}/{paramname}.out", encoding='utf-8', level=logging.INFO)
    
    #do the actual training
    res = train_L_layers(filename,datasetsize,L,r,weight_decay=weight_decay,epochs=epochs,
                        scheduler=MultiStepLR,milestones=[epochs-100], gamma=0.1, verbose=True,std=std)
    model,trainmse,weightdecay,learningrate,testmse = res

    logging.info("received results")
    
    #save Results
    np.save(f"{filename}/{paramname}testMSE",testmse.copy())
    np.save(f"{filename}/{paramname}trainMSEs",trainmse.clone().detach().cpu().numpy())
    np.save(f"{filename}/{paramname}weightdecays",weightdecay.clone().detach().cpu().numpy())
    np.save(f"{filename}/{paramname}learningrates",learningrate.clone().detach().cpu().numpy())
    torch.save(model.state_dict(), f"{filename}/{paramname}model.pt")
    
    logging.info("saved results")