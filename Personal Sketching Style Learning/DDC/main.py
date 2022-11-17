#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
import argparse
from ast import arg
from email.policy import default
from fileinput import close
import warnings
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tqdm import tnrange
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
#learning_rate = 1e-3
learning_rate = 1e-4
L2_DECAY = 5e-4
MOMENTUM = 0.9

def step_decay(epoch, learning_rate):
    """
    Schedule step decay of learning rate with epochs.
    """
    initial_learning_rate = learning_rate
    drop = 0.8
    epochs_drop = 10.0
    learning_rate = initial_learning_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

    return learning_rate

def main():
    """
    This method puts all the modules together to train DDCNet for image
    classification. It uses a MMD loss in the last classification layer for
    domain adaptation.

    Paper: https://arxiv.org/abs/1412.3474
    """
    
    parser = argparse.ArgumentParser(description="domain adaptation w MMD")

    parser.add_argument("--epochs", default=50, type=int,help="number of training epochs")

    parser.add_argument("--batch_size_source", default=128, type=int,help="batch size of source data")

    parser.add_argument("--batch_size_target", default=128, type=int,help="batch size of target data")

    parser.add_argument("--name_source", default="source", type=str,help="name of source dataset (default amazon)")

    parser.add_argument("--name_tgttrain", default="target_train_fsl5", type=str,help="name of source dataset (default webcam)")

    parser.add_argument("--name_tgttest", default="target_test_fsl5", type=str,help="name of source dataset (default webcam)")

    parser.add_argument("--num_classes", default=374, type=int,help="no. classes in dataset (default 374)")

    parser.add_argument("--load_model", default="logs/pretrain/10ResNet18.pth", type=str,help="load pretrained model (default None)")

    parser.add_argument("--adapt_domain", action='store_true',help="argument to compute coral loss (default False)")

    parser.add_argument("--target_supervised", default=False, type=bool,help="using labels of target domain when cross-training")

    parser.add_argument("--result_path",default="logs/sketch",type=str,help="save test result")
    args = parser.parse_args()

    # create dataloaders
    print("creating source/target dataloaders...")
    print("source data:",args.name_source)
    print("target train data:", args.name_tgttrain)
    print("target test data:", args.name_tgttest)

    source_loader = get_sketch_dataloader(name_dataset = args.name_source,batch_size = args.batch_size_source)  
    tgttrain_loader = get_sketch_dataloader(name_dataset = args.name_tgttrain,batch_size = args.batch_size_target)                                
    tgttest_loader = get_sketch_dataloader(name_dataset = args.name_tgttest,batch_size = args.batch_size_target) 

    # define DDCNet model
    model = DDCNet(num_classes=args.num_classes)

    # move to CUDA if available
    if CUDA:
        model = model.cuda()
        print("using cuda...")

    # load pre-trained model or pre-trained AlexNet
    if args.load_model is not None:
        load_model(model, args.load_model) # contains path to model params

    print("model type:", type(model))

    # store statistics of train/test
    training_statistic = []
    testing_s_statistic = []
    testing_v_statistic=[]
    testing_t_statistic = []

    # start training over epochs
    print("adapt domain:", args.adapt_domain)
    print("target supervised:",args.target_supervised)
    print("running training for {} epochs...".format(args.epochs))
    supervised=args.target_supervised

    for epoch in tnrange(0, args.epochs):

        log_interval = 10
        LEARNING_RATE = step_decay(epoch, learning_rate)
        print("Current learning rate:", LEARNING_RATE)

        optimizer = torch.optim.SGD([
            {"params": model.sharedNetwork.parameters()},
            {"params": model.bottleneck.parameters(),  "lr":LEARNING_RATE},
            {"params": model.fc.parameters(), "lr":LEARNING_RATE},
        ], lr=LEARNING_RATE/10, momentum=MOMENTUM, weight_decay=L2_DECAY)

        # compute lambda value from paper (eq 6)
        if args.adapt_domain:
            lambda_factor = (epoch+1)/args.epochs # adaptation (w/ coral loss)
        else:
            lambda_factor = 0 # no adaptation (w/o coral loss)

        if epoch==0:
            test_tgttrain = test(model, tgttrain_loader, epoch, CUDA)
            test_tgttest = test(model, tgttest_loader, epoch, CUDA)
            print("[Test Target_train]: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                epoch+1,
                test_tgttrain['average_loss'],
                test_tgttrain['correct_class'],
                test_tgttrain['total_elems'],
                test_tgttrain['accuracy %'],
            ))

            print("[Test Target_test]: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                epoch+1,
                test_tgttest['average_loss'],
                test_tgttest['correct_class'],
                test_tgttest['total_elems'],
                test_tgttest['accuracy %'],
            ))    

        # run batch trainig at each epoch (returns dictionary with epoch result)    
        result_train = train(model, source_loader,tgttrain_loader, optimizer, epoch+1, lambda_factor, CUDA, supervised)

        # print log values
        print("[EPOCH] {}: Classification loss: {:.6f}, DDC loss: {:.6f}, Total_Loss: {:.6f}".format(
                epoch+1,
                sum(row['classification_loss'] / row['total_steps'] for row in result_train),
                sum(row['ddc_loss'] / row['total_steps'] for row in result_train),
                sum(row['total_loss'] / row['total_steps'] for row in result_train),
            ))

        training_statistic.append(result_train)

        # test classification accuracy on both datasets
        test_source = test(model, source_loader, epoch, CUDA)
        test_tgttrain = test(model, tgttrain_loader, epoch, CUDA)
        test_tgttest = test(model, tgttest_loader, epoch, CUDA)
        testing_s_statistic.append(test_source)
        testing_v_statistic.append(test_tgttrain)
        testing_t_statistic.append(test_tgttest)

        
        print("[Test Source]: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                epoch+1,
                test_source['average_loss'],
                test_source['correct_class'],
                test_source['total_elems'],
                test_source['accuracy %'],
            ))


        print("[Test Target]: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                epoch+1,
                test_tgttrain['average_loss'],
                test_tgttrain['correct_class'],
                test_tgttrain['total_elems'],
                test_tgttrain['accuracy %'],
        ))

        print("[Test Target_test]: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                epoch+1,
                test_tgttest['average_loss'],
                test_tgttest['correct_class'],
                test_tgttest['total_elems'],
                test_tgttest['accuracy %'],
        ))

        f=open(args.result_path+"/txts/"+"result"+args.name_tgttrain.strip().split("/")[2]+".txt",'a')
        f.write("Epoch:"+str(epoch+1)+", target_supervised:"+str(args.target_supervised)+"\n")
        f.write("Test_source:"+"avg_loss:"+str(test_source["average_loss"])+",accuracy:"+str(test_source['correct_class'])+"/"+str(test_source['total_elems'])+","+str(round(test_source['correct_class']/test_source['total_elems'],3))+"\n")
        f.write("Test_tgttrain:"+"avg_loss:"+str(test_tgttrain["average_loss"])+",accuracy:"+str(test_tgttrain['correct_class'])+"/"+str(test_tgttrain['total_elems'])+","+str(round(test_tgttrain['correct_class']/test_tgttrain['total_elems'],3))+'\n')
        f.write("Test_tgttest:"+"avg_loss:"+str(test_tgttest["average_loss"])+",accuracy:"+str(test_tgttest['correct_class'])+"/"+str(test_tgttest['total_elems'])+","+str(round(test_tgttest['correct_class']/test_tgttest['total_elems'],3))+'\n')
        f.close()


    # save log results
    if args.adapt_domain:
        print("saving training with adaptation...")
        save_log(training_statistic, args.result_path+'/out/'+args.name_tgttrain.strip().split("/")[2]+"_"+str(int(args.target_supervised))+'1_training_statistic.pkl')
        save_log(testing_s_statistic, args.result_path+'/out/'+args.name_tgttrain.strip().split("/")[2]+"_"+str(int(args.target_supervised))+'1_testing_s_statistic.pkl')
        save_log(testing_v_statistic, args.result_path+'/out/'+args.name_tgttrain.strip().split("/")[2]+"_"+str(int(args.target_supervised))+'1_testing_v_statistic.pkl')
        save_log(testing_t_statistic, args.result_path+'/out/'+args.name_tgttrain.strip().split("/")[2]+"_"+str(int(args.target_supervised))+'1_testing_t_statistic.pkl')
        save_model(model, args.result_path+'/out/'+args.name_tgttrain.strip().split("/")[2]+"_"+str(int(args.target_supervised))+'1_checkpoint.tar')

    else:
        print("saving training without adaptation...")
        save_log(training_statistic, args.result_path+'/out/'+args.name_tgttrain.strip().split("/")[2]+"_"+str(int(args.target_supervised))+'0_training_statistic.pkl')
        save_log(testing_s_statistic, args.result_path+'/out/'+args.name_tgttrain.strip().split("/")[2]+"_"+str(int(args.target_supervised))+'0_testing_s_statistic.pkl')
        save_log(testing_v_statistic, args.result_path+'/out/'+args.name_tgttrain.strip().split("/")[2]+"_"+str(int(args.target_supervised))+'0_testing_v_statistic.pkl')
        save_log(testing_t_statistic, args.result_path+'/out/'+args.name_tgttrain.strip().split("/")[2]+"_"+str(int(args.target_supervised))+'0_testing_t_statistic.pkl')
        save_model(model, args.result_path+'/out/'+args.name_tgttrain.strip().split("/")[2]+"_"+str(int(args.target_supervised))+'0_checkpoint.tar')    

if __name__ == '__main__':
    main()
