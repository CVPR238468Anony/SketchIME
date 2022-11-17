#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#TODO: get curves
# TODO: try different combinations among datasets

from __future__ import division
import argparse
import warnings
from tqdm import tnrange
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
from torch.autograd import Variable
warnings.filterwarnings("ignore")

from train import train
from test import test
from utils import save_log, save_model, load_model
from dataloader import get_sketch_dataloader
from model import  ResNet18, AdversarialNetwork, baseNetwork
import network
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# set model hyperparameters (paper page 5)
CUDA = True if torch.cuda.is_available() else False
#CUDA = False
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
# BATCH_SIZE = [32, 32] # batch_s, batch_t [128, 56]
# EPOCHS = 1
def main():
    """
    This method puts all the modules together to train a neural network
    classifier using CORAL loss.

    Reference: https://arxiv.org/abs/1607.01719
    """
    parser = argparse.ArgumentParser(description="domain adaptation w CORAL")

    parser.add_argument("--epochs", default=50, type=int,help="number of training epochs")

    parser.add_argument("--batch_size_source", default=16, type=int,help="batch size of source data") 

    parser.add_argument("--batch_size_target", default=16, type=int,help="batch size of target data")

    parser.add_argument("--name_source", default="source", type=str,help="name of source dataset (default amazon)")  

    parser.add_argument("--name_tgttrain", default="target_train_fsl5", type=str,help="name of source dataset (default webcam)") 

    parser.add_argument("--name_tgttest", default="target_test_fsl5", type=str,help="name of source dataset (default webcam)")

    parser.add_argument("--num_classes", default=374, type=int,help="no. classes in dataset (default 374)") #一共374类

    parser.add_argument("--load_model", default="logs/CDAN/pretrain/10ResNet18.pth", type=str,help="load pretrained model (default None)")
    
    parser.add_argument("--target_supervised", default=False, type=bool,help="using labels of target domain when cross-training")

    parser.add_argument("--result_path",default="logs/CDAN/sketch",type=str,help="save the result")

    parser.add_argument("--entropy",default=False,type=bool,help="CDAN or CDAN+E")

    args = parser.parse_args()

    # create dataloaders (Amazon as source and Webcam as target)
    print("creating source/target dataloaders...")
    print("source data:", args.name_source)
    print("target train data:", args.name_tgttrain)
    print("target test data:", args.name_tgttest)

    source_loader = get_sketch_dataloader(name_dataset = args.name_source,batch_size = args.batch_size_source)

    tgttrain_loader = get_sketch_dataloader(name_dataset = args.name_tgttrain,batch_size = args.batch_size_target)

    tgttest_loader = get_sketch_dataloader(name_dataset = args.name_tgttest,batch_size = args.batch_size_target)

    # define  network
    bottleneck_dim = 256
    model = baseNetwork(num_classes=args.num_classes,bottleneck_dim=bottleneck_dim)
    ad_net= AdversarialNetwork(bottleneck_dim*args.num_classes,1024)
    model.train(True)
    ad_net.train(True)
    # define optimizer pytorch: https://pytorch.org/docs/stable/optim.html
    # specify learning rates per layers:
    # 10*learning_rate for last two fc layers according to paper
    optimizer = torch.optim.SGD([
        {"params": model.sharedNetwork.parameters()},
        {"params": model.fc8.parameters(), "lr":10*LEARNING_RATE},
        {"params":ad_net.parameters(), "lr_mult": 10, 'decay_mult': 2}
    ], lr=LEARNING_RATE, momentum=MOMENTUM)

    # move to CUDA if available
    if CUDA:
        model = model.cuda()
        print("using cuda...")

    # load pre-trained model or pre-trained AlexNet
    if args.load_model is not None:
        load_model(model, args.load_model) # contains path to model params

    print("model type:", type(model))

    # store statistics of train/test
    training_s_statistic = []
    testing_s_statistic = []
    testing_t_statistic = []

    # start training over epochs
    #print("running training for {} epochs...".format(args.epochs))
    print("target supervised:", args.target_supervised)
    print("running training for {} epochs...".format(args.epochs))
    for epoch in range(0, args.epochs):
        # compute lambda value from paper (eq 6)
        lambda_factor = (epoch+1)/args.epochs

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
        result_train = train(model, ad_net, tgttrain_loader,tgttest_loader, optimizer, epoch+1, lambda_factor, CUDA, args.target_supervised,args.entropy)

        # print log values
        print("[EPOCH] {}: Classification: {:.6f}, CDAN loss: {:.6f}, Total_Loss: {:.6f}".format(
                epoch+1,
                sum(row['classification_loss'] / row['total_steps'] for row in result_train),
                sum(row['cdan_loss'] / row['total_steps'] for row in result_train),
                sum(row['total_loss'] / row['total_steps'] for row in result_train),
            ))

        training_s_statistic.append(result_train)

        # perform testing simultaneously: classification accuracy on both dataset
        test_source = test(model, source_loader, epoch, CUDA)
        test_tgttrain = test(model, tgttrain_loader, epoch, CUDA)
        test_tgttest = test(model, tgttest_loader, epoch, CUDA)
        testing_s_statistic.append(test_source)
        testing_t_statistic.append(test_tgttrain)
        testing_t_statistic.append(test_tgttest)

        print("[Test Source]: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                epoch+1,
                test_source['average_loss'],
                test_source['correct_class'],
                test_source['total_elems'],
                test_source['accuracy %'],
            ))

        print("[Test Tgttrain]: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
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
        f.write("Test_source:"+"avg_loss:"+str(test_source["average_loss"])+",accuracy:"+str(test_source['correct_class'])+"/"+str(test_source['total_elems'])+","+str(round(test_source['correct_class'].item()/test_source['total_elems'],3))+"\n")
        f.write("Test_tgttrain:"+"avg_loss:"+str(test_tgttrain["average_loss"])+",accuracy:"+str(test_tgttrain['correct_class'])+"/"+str(test_tgttrain['total_elems'])+","+str(round(test_tgttrain['correct_class'].item()/test_tgttrain['total_elems'],3))+'\n')
        f.write("Test_tgttest:"+"avg_loss:"+str(test_tgttest["average_loss"])+",accuracy:"+str(test_tgttest['correct_class'])+"/"+str(test_tgttest['total_elems'])+","+str(round(test_tgttest['correct_class'].item()/test_tgttest['total_elems'],3))+'\n')
        f.close()
        

    # save results
    if not args.entropy:        
        print("saving results...")
        save_log(training_s_statistic, args.result_path+'/out/'+args.name_tgttrain.strip().split('/')[2]+str(args.target_supervised)+'sup_no_adaptation_training_s_statistic.pkl')
        save_log(testing_s_statistic, args.result_path+'/out/'+args.name_tgttrain.strip().split('/')[2]+str(args.target_supervised)+'sup_no_adaptation_testing_s_statistic.pkl')
        save_log(testing_t_statistic, args.result_path+'/out/'+args.name_tgttrain.strip().split('/')[2]+str(args.target_supervised)+'sup_no_adaptation_testing_t_statistic.pkl')
        save_log(training_s_statistic, args.result_path+'/out/'+args.name_tgttrain.strip().split('/')[2]+str(args.target_supervised)+'sup_training_s_statistic.pkl')
        save_log(testing_s_statistic, args.result_path+'/out/'+args.name_tgttrain.strip().split('/')[2]+str(args.target_supervised)+'sup_testing_s_statistic.pkl')
        save_log(testing_t_statistic, args.result_path+'/out/'+args.name_tgttrain.strip().split('/')[2]+str(args.target_supervised)+'sup_testing_t_statistic.pkl')
        save_model(model, args.result_path+'/out/'+args.name_tgttrain.strip().split('/')[2]+str(args.target_supervised)+'checkpoint.tar')
    else:
        print("saving results...")
        save_log(training_s_statistic, args.result_path+'/out/'+args.name_tgttrain.strip().split('/')[2]+str(args.target_supervised)+'sup_no_adaptation_training_s_statistic.pkl')
        save_log(testing_s_statistic, args.result_path+'/out/'+args.name_tgttrain.strip().split('/')[2]+str(args.target_supervised)+'sup_no_adaptation_testing_s_statistic.pkl')
        save_log(testing_t_statistic, args.result_path+'/out/'+args.name_tgttrain.strip().split('/')[2]+str(args.target_supervised)+'sup_no_adaptation_testing_t_statistic.pkl')
        save_log(training_s_statistic, args.result_path+'/out/'+args.name_tgttrain.strip().split('/')[2]+str(args.target_supervised)+'sup_training_s_statistic.pkl')
        save_log(testing_s_statistic, args.result_path+'/out/'+args.name_tgttrain.strip().split('/')[2]+str(args.target_supervised)+'sup_testing_s_statistic.pkl')
        save_log(testing_t_statistic, args.result_path+'/out/'+args.name_tgttrain.strip().split('/')[2]+str(args.target_supervised)+'sup_testing_t_statistic.pkl')
        save_model(model, args.result_path+'/out/'+args.name_tgttrain.strip().split('/')[2]+str(args.target_supervised)+'checkpoint.tar')




if __name__ == '__main__':
    main()
