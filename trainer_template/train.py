import torch
import os
import sys
import time
from datetime import timedelta


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
ROOT_PATH = os.path.abspath(os.curdir)


def print_progress(index, total, fi="", last=""):
    percent = ("{0:.1f}").format(100 * ((index) / total))
    fill = int(30 * (index / total))
    spec_char = ["\x1b[1;31;40m╺\x1b[0m",
                 "\x1b[1;36;40m━\x1b[0m", "\x1b[1;37;40m━\x1b[0m"]
    bar = spec_char[1]*(fill-1) + spec_char[0] + spec_char[2]*(30-fill)
    if fill == 30:
        bar = spec_char[1]*fill

    percent = " "*(5-len(str(percent))) + str(percent)

    if index == total:
        print(fi + " " + bar + " " + percent + "% " + last)
    else:
        print(fi + " " + bar + " " + percent + "% " + last, end="\r")


class Trainer:


    def __init__(self, model, optimizer=None, criterion=None, scheduler=None, **kwargs):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if "device" in kwargs:
            self.device = kwargs["device"]
        
        self.model = model 
        self.model = self.model.to(self.device)
        self.optimizer = optimizer 
        self.scheduler = scheduler 
        self.criterion = criterion 

        self.train_loss = []
        self.valid_loss = []
        self.train_acc = []
        self.valid_acc = []


    def load_checkpoint(self, path):
        params = torch.load(path)
        self.model.load_state_dict(params["model"])

        if self.optimizer is not None:
            self.optimizer.load_state_dict(params["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(params["scheduler"])

        self.train_loss = params["train_loss"]
        self.valid_loss = params["valid_loss"]
        self.train_acc = params["train_acc"]
        self.valid_acc = params["valid_acc"]

        print("[+] Model load successful")


    def save_checkpoint(self, path):
        params = {
            "train_loss": self.train_loss,
            "valid_loss": self.valid_loss,
            "train_acc": self.train_acc,
            "valid_acc": self.valid_acc,
            "model": self.model.state_dict()
        }
        if self.optimizer is not None:
            params["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            params["scheduler"] = self.scheduler.state_dict()
        
        torch.save(params, path)


    def accuracy_compute(self):
        pass 


    def loss_compute(self):
        pass 


    def train_step(self, dataloader):
        self.model.train()
        loss_his = []
        acc_his = []
        N = len(dataloader)
        for idx, (X, y) in enumerate(dataloader):
            start_time = time.time()
            self.optimizer.zero_grad()
            X = X.to(self.device)
            y = y.to(self.device)
             
            # Logic here

            loss = self.accuracy_compute()
            acc = self.accuracy_compute()


            
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()


            loss_his.append(loss.item())
            acc_his.append(acc)
            
            total_time = round(time.time() - start_time, 1)
            rem_time = total_time*(N - idx - 1)
            eta_time = timedelta(seconds=int(rem_time))
            time_string = f"\x1b[1;31;40m{total_time}s/step\x1b[0m eta\x1b[1;36;40m {eta_time}\x1b[0m"

            print_progress(
                idx+1, 
                len(self.train_loader), 
                last=f"Time: {time_string} Loss: {round(loss.item(), 5)} Acc: {acc}",
                fi=f"Train batch {' '*(len(str(N))-len(str(idx+1)))}{idx+1}/{len(self.train_loader)}")
        loss_his = sum(loss_his) / len(loss_his)
        acc_his = sum(acc_his) / len(acc_his)
        self.train_loss.append(loss_his)
        self.train_acc.append(acc_his)
            

    def valid_step(self, dataloader):
        self.model.eval()
        loss_his = []
        acc_his = []
        N = len(dataloader)
        for idx, (X, y) in enumerate(dataloader):
            start_time = time.time()
            X = X.to(self.device)
            y = y.to(self.device)

            loss = self.accuracy_compute()
            acc = self.accuracy_compute()

            loss_his.append(loss.item())
            acc_his.append(acc)

            total_time = round(time.time() - start_time, 1)
            rem_time = total_time*(N - idx - 1)
            eta_time = timedelta(seconds=int(rem_time))
            time_string = f"\x1b[1;31;40m{total_time}s/step\x1b[0m eta\x1b[1;36;40m {eta_time}\x1b[0m"

            print_progress(
                idx+1, 
                len(self.train_loader), 
                last=f"Time: {time_string} Loss: {round(loss.item(), 5)} Acc: {acc}",
                fi=f"Valid batch {' '*(len(str(N))-len(str(idx+1)))}{idx+1}/{len(self.train_loader)}")
            
        loss_his = sum(loss_his) / len(loss_his)
        acc_his = sum(acc_his) / len(acc_his)
        self.train_loss.append(loss_his)
        self.train_acc.append(acc_his)


    def fit(self, train_loader, valid_loader=None, epochs=5, verbose=False, checkpoint="./checkpoint/checkpoint_v1.pt"):
        self.verbose = verbose

        for epoch in range(1, epochs+1):
            start_time = time.time()
            print(f"Epoch: {epoch}")
            try:
                self.train_step(train_loader)
            except KeyboardInterrupt:
                sys.exit()
            
            try:
                self.valid_step(valid_loader)
            except KeyboardInterrupt:
                sys.exit()
            
            self.save_checkpoint(checkpoint)

            total_time = round(time.time() - start_time, 1)
            
            train_loss = round(self.train_loss[-1], 5)
            valid_loss = round(self.valid_loss[-1], 5)
            train_acc = round(self.train_acc[-1], 2)
            valid_acc = round(self.valid_acc[-1], 2)

            print(f"\t=> Train loss: {train_loss} - Valid loss: {valid_loss} - Train acc: {train_acc} - Valid acc: {valid_acc} - Time: {timedelta(seconds=int(total_time))}/step\n")
            
