import torch
<<<<<<< HEAD
import pandas as pd
import os
from tqdm import tqdm
=======
import os
>>>>>>> a7b804288cbc931c7b24addf1ae52ef9f6f3757f
import sys
import time
from datetime import timedelta

<<<<<<< HEAD
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Trainer:
=======

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


>>>>>>> a7b804288cbc931c7b24addf1ae52ef9f6f3757f
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

<<<<<<< HEAD
=======

>>>>>>> a7b804288cbc931c7b24addf1ae52ef9f6f3757f
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

<<<<<<< HEAD
=======

>>>>>>> a7b804288cbc931c7b24addf1ae52ef9f6f3757f
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

<<<<<<< HEAD
    def accuracy_compute(self, pred, labels):
        return 0

    def loss_compute(self, logits, labels):
        loss = self.criterion(logits, labels)
        return loss
    
=======

    def accuracy_compute(self):
        pass 


    def loss_compute(self):
        pass 


>>>>>>> a7b804288cbc931c7b24addf1ae52ef9f6f3757f
    def train_step(self, dataloader):
        self.model.train()
        loss_his = []
        acc_his = []
        N = len(dataloader)
<<<<<<< HEAD
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
=======
        for idx, (X, y) in enumerate(dataloader):
            start_time = time.time()
            self.optimizer.zero_grad()
            X = X.to(self.device)
            y = y.to(self.device)
             
            # Logic here

            loss = self.accuracy_compute()
            acc = self.accuracy_compute()


            
>>>>>>> a7b804288cbc931c7b24addf1ae52ef9f6f3757f
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

<<<<<<< HEAD
            loss_his.append(loss.item())
            acc_his.append(acc)
            total_time = round(time.time() - start_time, 1)
            rem_time = total_time*(N - idx - 1)
            eta_time = timedelta(seconds=int(rem_time))
            train_tqdm.set_description('lr: {0}, Training'.format(self.optimizer.param_groups[0]['lr']))
            train_tqdm.set_postfix(loss=loss.item())
            
=======

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
>>>>>>> a7b804288cbc931c7b24addf1ae52ef9f6f3757f
        loss_his = sum(loss_his) / len(loss_his)
        acc_his = sum(acc_his) / len(acc_his)
        self.train_loss.append(loss_his)
        self.train_acc.append(acc_his)
<<<<<<< HEAD
    
    @torch.no_grad()
=======
            

>>>>>>> a7b804288cbc931c7b24addf1ae52ef9f6f3757f
    def valid_step(self, dataloader):
        self.model.eval()
        loss_his = []
        acc_his = []
        N = len(dataloader)
<<<<<<< HEAD
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
=======
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
>>>>>>> a7b804288cbc931c7b24addf1ae52ef9f6f3757f
        self.verbose = verbose

        for epoch in range(1, epochs+1):
            start_time = time.time()
            print(f"Epoch: {epoch}")
            try:
                self.train_step(train_loader)
<<<<<<< HEAD
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
=======
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
            
>>>>>>> a7b804288cbc931c7b24addf1ae52ef9f6f3757f
