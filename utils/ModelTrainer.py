


#model, device, train_loader, optimizer, epoch, log_interval=100, verbose=True



class ModelTrainer:
    #Falta passar paràmetres model i learning rate
    def __init__(self,train_load,optim,loss_function,epochs=10):
        self.epochs=epochs
        self.train_loader=train_load
        self.optimizer=optim
        self.loss_fn=loss_function

    def train(self,model,device):
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # !!!
        print("Trainable Parameters ", pytorch_total_params)

        model.train()

        loss_v = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
        
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = model(data)
            output = torch.squeeze(output)
            loss=F.cross_entropy(output, target, reduction='mean')
            loss.backward()
            optimizer.step() 
            if batch_idx % log_interval == 0 and verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Average: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), loss.item()/ len(data)))
            loss_v += loss.item()

        loss_v /= len(train_loader.dataset)
        #print('\nTrain set: Average loss: {:.4f}\n'.format(loss_v))
    
        return loss_v