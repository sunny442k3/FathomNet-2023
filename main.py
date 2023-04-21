import torch, gc
import torchvision.transforms as transforms
import pandas as pd
from transformers import AutoFeatureExtractor, AutoModel, AutoImageProcessor
import argparse
from dataset import FathomDataset
from torch.utils.data import Dataset, DataLoader
from utils import *
from trainer_template.train import Trainer
from models import Query2label
from torchvision.transforms import RandAugment
from losses import AsymmetricLoss
from torch import nn, optim

if __name__=='__main__':
    torch.cuda.empty_cache()
    gc.collect()
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_labels', type=str, default='/kaggle/input/data-csv/train_with_labels.csv')
    parser.add_argument('--train_img_root', type=str, help='path to train image dataset', default='./kaggle/input/fathomnettrain/images/train')
    parser.add_argument('--eval_img_root', type=str, help='path to eval image dataset', default='/kaggle/input/fathomneteval/images/train')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--model_name', default='microsoft/resnet-50')
    parser.add_argument('--model_path', default='./checkpoint.pth', type=str)
    parser.add_argument('--num_classes', default=290)
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--image_size', default=256, type=int,
                        metavar='N', help='input image size (default: 448)')
    parser.add_argument('--thre', default=0.8, type=float,
                        metavar='N', help='threshold value')
    parser.add_argument('-b', '--batch-size', default=56, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--decoder_dim', default=768, type=int)
    parser.add_argument('--gamma_neg', default=4, type=float)
    parser.add_argument('--gamma_pos', default=0, type=float)
    parser.add_argument('--epochs', type=int, default=40)

    args = parser.parse_args()
    train_root = str(args.train_img_root)
    eval_root = str(args.eval_img_root)
    train_csv = pd.read_csv()
    transform = transform =  transforms.Compose([transforms.Resize((args.image_size, args.image_size)), CutoutPIL(cutout_factor=0.5), RandAugment(), transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    dataset = FathomDataset(train_root, train_csv.copy(), transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    backbone = AutoModel.from_pretrained(args.model_name)
    model = Query2label(backbone, args.num_classes, args.decoder_dim, spatial_dim=768)
    criterion = AsymmetricLoss(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos)
    steps_per_epoch = len(dataloader)
    lr = args.lr
    optimizer = optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs,
                                            pct_start=0.2)

    trainer = Trainer(model, optimizer, criterion, scheduler)
    trainer.fit(dataloader)
    trainer.load_checkpoint(args.checkpoint)