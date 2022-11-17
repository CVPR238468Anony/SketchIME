#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
import argparse
import warnings
from tqdm import tnrange
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
from torch.autograd import Variable
warnings.filterwarnings("ignore")

from train import train
from test import test
from loss import DDC_loss
from utils import save_log, save_model, load_model
from dataloader import get_sketch_dataloader
from model import DDCNet

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# set model hyperparameters (paper page 5)
CUDA = True if torch.cuda.is_available() else False
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

def main():
    """
    This method puts all the modules together to train DeepCORAL for image
    classification. It uses a CORAL loss in the last classification layer for
    domain adaptation.

    Paper: https://arxiv.org/abs/1607.01719
    """
    parser = argparse.ArgumentParser(description="domain adaptation w CORAL")

    parser.add_argument("--epochs", default=10, type=int, help="number of training epochs")

    parser.add_argument("--batch_size_source", default=128, type=int, help="batch size of source data")
    
    parser.add_argument("--name_source", default="source", type=str, help="name of source dataset (default source)")

    parser.add_argument("--name_target", default="target_train", type=str, help="name of target dataset (default target_train)")

    parser.add_argument("--num_classes", default=374, type=int, help="no. classes in dataset (default 31)")               
                        
    args = parser.parse_args()

    # create dataloaders
    print("creating source dataloaders...")
    print("source data:", args.name_source)
    
    source_loader1 = get_sketch_dataloader(name_dataset = args.name_source,batch_size = args.batch_size_source)
    source_loader2 = get_sketch_dataloader(name_dataset = args.name_source,batch_size = args.batch_size_source)
                                      
    # define DeepCORAL model
    model = DDCNet(num_classes=args.num_classes)

    # define optimizer: https://pytorch.org/docs/stable/optim.html
    # specify learning rates per layers:
    # 10*learning_rate for last two fc layers according to paper
    optimizer = torch.optim.SGD([
        {"params": model.sharedNetwork.parameters()},
        {"params": model.bottleneck.parameters()},
        {"params": model.fc.parameters()},
    ], lr=LEARNING_RATE, momentum=MOMENTUM)

    # move to CUDA if available
    if CUDA:
        model = model.cuda()
        print("using cuda...")

    print("model type:", type(model))

    # store statistics of train/test
    training_statistic = []

    # start training over epochs
    print("running training for {} epochs...".format(args.epochs))
    for epoch in tnrange(0, args.epochs):
        # compute lambda value from paper (eq 6)
        lambda_factor = 0 # no adaptation (w/o coral loss)

        # run batch trainig at each epoch (returns dictionary with epoch result)
        result_train = train(model, source_loader1, source_loader2, optimizer, epoch+1, lambda_factor, CUDA, False)

        # print log values
        print("[EPOCH] {}: Classification loss: {:.6f}, Total_Loss: {:.6f}".format(
                epoch+1,
                sum(row['classification_loss'] / row['total_steps'] for row in result_train),
                sum(row['total_loss'] / row['total_steps'] for row in result_train),
            ))

        training_statistic.append(result_train)

        # test classification accuracy on both datasets
        test_source = test(model, source_loader2, epoch, CUDA)

        print("[Test Source]: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                epoch+1,
                test_source['average_loss'],
                test_source['correct_class'],
                test_source['total_elems'],
                test_source['accuracy %'],
            ))
        
        torch.save(model.state_dict(), "logs/pretrain/"+str(epoch+1)+"ResNet18.pth")
    # save log results
    print("saving training without adaptation...")
    save_log(training_statistic, 'logs/pretrain/no_adaptation_training_statistic.pkl')
    save_model(model, 'logs/pretrain/no_adaptation_training_checkpoint.tar')


if __name__ == '__main__':
    main()
