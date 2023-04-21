import torch
import pandas as pd
import os
from tqdm import tqdm
import sys
import time
from datetime import timedelta

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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

    def accuracy_compute(self, pred, labels):
        return 0

    def loss_compute(self, logits, labels):
        loss = self.criterion(logits, labels)
        return loss
    
    def train_step(self, dataloader):
        self.model.train()
        loss_his = []
        acc_his = []
        N = len(dataloader)
        train_tqdm = tqdm(dataloader)
        for idx, batch in enumerate(train_tqdm):
            image, labels = batch
            image = image.to(self.device)
            labels = labels.to(self.device)
            start_time = time.time()
            self.optimizer.zero_grad()
            logits = self.model(image)
            loss = self.loss_compute(logits, labels)
            with torch.no_grad():
                acc = self.accuracy_compute(logits, labels)
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            loss_his.append(loss.item())
            acc_his.append(acc)
            total_time = round(time.time() - start_time, 1)
            rem_time = total_time*(N - idx - 1)
            eta_time = timedelta(seconds=int(rem_time))
            train_tqdm.set_description('lr: {0}, Training'.format(self.optimizer.param_groups[0]['lr']))
            train_tqdm.set_postfix(loss=loss.item())
            
        loss_his = sum(loss_his) / len(loss_his)
        acc_his = sum(acc_his) / len(acc_his)
        self.train_loss.append(loss_his)
        self.train_acc.append(acc_his)
    
    @torch.no_grad()
    def valid_step(self, dataloader):
        self.model.eval()
        loss_his = []
        acc_his = []
        N = len(dataloader)
        with torch.no_grad():
            valid_tqdm = tqdm(dataloader, leave=True)
            for idx, batch in enumerate(valid_tqdm):
                image, labels = batch
                image = image.to(self.device)
                labels = labels.to(self.device)
                start_time = time.time()
                logits = self.model(image)
                loss = self.loss_compute(logits, labels)
                acc = self.accuracy_compute(logits, labels)

                loss_his.append(loss.item())
                acc_his.append(acc)

                total_time = round(time.time() - start_time, 1)
                rem_time = total_time*(N - idx - 1)
                eta_time = timedelta(seconds=int(rem_time))
                valid_tqdm.set_description('lr: {0}, Validation'.format(self.optimizer.param_groups[0]['lr']))
                valid_tqdm.set_postfix(loss=loss.item())
            
            loss_his = sum(loss_his) / len(loss_his)
            acc_his = sum(acc_his) / len(acc_his)
            self.train_loss.append(loss_his)
            self.train_acc.append(acc_his)

    def fit(self, train_loader, valid_loader=None, epochs=40, verbose=False, checkpoint="./checkpoint.pt"):
        self.verbose = verbose

        for epoch in range(1, epochs+1):
            start_time = time.time()
            print(f"Epoch: {epoch}")
            try:
                self.train_step(train_loader)
                train_loss = round(self.train_loss[-1], 5)
                train_acc = round(self.train_acc[-1], 2)
            except KeyboardInterrupt:
                sys.exit()
            if valid_loader is not None:
                try:
                    self.valid_step(valid_loader)
                    valid_loss = round(self.valid_loss[-1], 5)
                    valid_acc = round(self.valid_acc[-1], 2)
                    total_time = round(time.time() - start_time, 1)
                    print(f"\t=> Train loss: {train_loss} - Valid loss: {valid_loss} - Train acc: {train_acc} - Valid acc: {valid_acc} - Time: {timedelta(seconds=int(total_time))}/step\n")
                except KeyboardInterrupt:
                    sys.exit()
            else:
                total_time = round(time.time() - start_time, 1)
                print(f"\t=> Train loss: {train_loss} - Train acc: {train_acc} - Time: {timedelta(seconds=int(total_time))}/step\n")
            self.save_checkpoint(checkpoint)
        
    @torch.no_grad()
    def inference(self, dataloader):
        results = pd.DataFrame()
        categories = []
        id_files = []
        for img, id_file in tqdm(dataloader):
            img = img.to(self.device)
            logits = self.model(img)
            prob = torch.sigmoid(logits)
            pred = torch.argsort(prob, dim=-1, descending=True)[:, :20].squeeze().cpu()
            categories.extend(pred.tolist())
            id_files.extend(id_file)
        results['id'] = id_files
        results['categories'] = categories
        return results