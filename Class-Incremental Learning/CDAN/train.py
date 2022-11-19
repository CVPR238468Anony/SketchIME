#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from tqdm import tnrange
from torch.autograd import Variable
from loss import CDAN,DANN, Entropy
from model import calc_coeff


def train(model, ad_net, source_loader, target_loader, optimizer, epoch, lambda_factor, cuda=False,supervised=False,usentropy=False):
    """
    This method fits the network params one epoch at a time.
    Implementation based on:
    https://github.com/SSARCandy/DeepCORAL/blob/master/main.py
    """


    results = [] # list to append loss values at each epoch

    # first cast into an iterable list the data loaders
    # data_source: (batch_size, channels, height, width)
    # data_target: (batch_size)
    # source[0][1][0].size() --> torch.Size([128, 3, 224, 224])

    # memory leakage
    # print("memory leak")
    source = list(enumerate(source_loader))
    # print(source)
    target = list(enumerate(target_loader))
    # train_steps = max(len(source), len(target))
    train_steps = max(len(source), len(target))

    # start batch training
    model.train()
    ad_net.train()
    for batch_idx in range(train_steps):
        # fetch data in batches
        # _, source_data -> torch.Size([128, 3, 224, 224]), labels -> torch.Size([128])
        _, (source_data, source_label) = source[batch_idx % len(source)]
        _, (target_data, target_label) = target[batch_idx % len(target)]

        if cuda:
            # move to device
            source_data = source_data.cuda()
            source_label = source_label.cuda()
            target_data = target_data.cuda()
            target_label = target_label.cuda()

        # create pytorch variables, the variables and functions build a dynamic graph of computation
        # source_data, source_label = Variable(source_data), Variable(source_label)
        # target_data, target_label = Variable(target_data), Variable(target_label)


        # reset to zero optimizer gradients
        optimizer.zero_grad()

        # do a forward pass through network (recall DeepCORAL outputs source, target activation maps)
        src_features, src_ouputs = model(source_data)
        tgt_features, tgt_ouputs = model(target_data)
        features = torch.cat((src_features, tgt_features), dim=0)
        outputs = torch.cat((src_ouputs, tgt_ouputs), dim=0)
        # print("<traning>feature size: {}, output size: {}".format(features.size(), outputs.size()))
        softmax_out = nn.Softmax(dim=1)(outputs)
        entropy = Entropy(softmax_out)
        if usentropy:
            #print("useentropy:",usentropy)
            transfer_loss = CDAN([features, softmax_out], ad_net, entropy, calc_coeff(batch_idx), None) #CDAN+E
        else:
            #print("useentropy:",usentropy)  
            transfer_loss = CDAN([features, softmax_out], ad_net)  #CDAN

        # compute losses (classification and coral loss)
        if supervised:
            #print("supervised:",supervised)
            classification_loss = torch.nn.functional.cross_entropy(src_ouputs, source_label)+torch.nn.functional.cross_entropy(tgt_ouputs, target_label)# supervised learning
        else:
            #print("supervised:",supervised)
            classification_loss = torch.nn.functional.cross_entropy(src_ouputs, source_label) # unsupervised learning

        #transfer_loss = CDAN([features, softmax_out], ad_net)  #CDAN
        # transfer_loss = DANN(features,ad_net)

        # print(type(transfer_loss))

        # compute total loss (equation 6 paper)
        total_loss = classification_loss + lambda_factor*transfer_loss
        # total_loss = classification_loss


        # compute gradients of network (backprop in pytorch)
        total_loss.backward()

        # update weights of network
        optimizer.step()

        # append results for each batch iteration as dictionaries
        results.append({
            'epoch': epoch,
            'step': batch_idx + 1,
            'total_steps': train_steps,
            'lambda': lambda_factor,
            'cdan_loss': transfer_loss.item(), # coral_loss.data[0],
            'classification_loss': classification_loss.item(),  # classification_loss.data[0],
            'total_loss': total_loss.item() # total_loss.data[0]
        })

        # print training info
        print("Train Epoch: {:2d} [{:2d}/{:2d}], Lambda value: {:.4f}, Classification loss: {:.6f}, CDAN loss: {:.6f}, Total_Loss: {:.6f}".format(
                  epoch,
                  batch_idx + 1,
                  train_steps,
                  lambda_factor,
                  classification_loss.item(), # classification_loss.data[0],
                  transfer_loss.item(), # coral_loss.data[0],
                  total_loss.item() # total_loss.data[0]
              ))
    return results
