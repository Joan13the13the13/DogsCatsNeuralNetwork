from tqdm import tqdm
from utils.Utils import train,test
import numpy as np
import torch
import torch.nn.functional as F

#model, device, train_loader, optimizer, epoch, log_interval=100, verbose=True



class ModelTrainer:
    #Falta passar paràmetres model i learning rate
    def __init__(self,train_load,test_load,optim,loss_function,epochs=10):
        self.epochs=epochs
        self.train_loader=train_load
        self.test_loader=test_load
        self.optimizer=optim
        self.loss_fn=loss_function

    def train_test(self,model,device):
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # !!!
        print("Trainable Parameters: ", pytorch_total_params)

        train_l = np.zeros((self.epochs))
        test_l = np.zeros((self.epochs))

        
        pbar = tqdm(range(1,self.epochs+1))
        for epoch in pbar:
            train_l[epoch-1]=train(model,device,self.train_loader,self.optimizer)
            test_l[epoch-1]=test(model,device,self.test_loader)

        #plot loss function????
        return (train_l,test_l)

        