#!/bin/sh

# python train.py --backbone_name resnet18 --model_name my --dataset_type cl_shot --optim sgd --opt opt2 \
# --loss_type ce --no_order --seed 1993 --way 5 --shot 5 --session 9 --lr 0.02 --lr-scheduler cos --batch_size 128 \
# --gpu-ids 1  --val  --data_dir ./datasets \
# --model_path ./log/CIFAR100_my_None_0.02/model_ls_0_CIFAR100.pth

#python train.py --backbone_name resnet18 --model_name my --dataset_name mini-imagenet --dataset_type cl_shot \
#--optim sgd --opt opt2 --loss_type ce --no_order --seed 1993 --way 5 --shot 5 --session 9 --lr 0.1 --lr-scheduler cos \
#--batch_size 128 --gpu-ids 0 --val --data_dir $your_path \
#--model_path ./pre/model_session_0_mini-imagenet.pth

#python train.py --backbone_name resnet18 --model_name my --pretrained --dataset_name cub --dataset_type cl_shot \
#--optim sgd --opt opt2 --loss_type ce --no_order --seed 1993 --way 10 --shot 5 --session 11 --lr 0.1 --lr-scheduler cos \
#--batch_size 128 --gpu-ids 0 --base_class 100 --val --data_dir $your_path \
#--model_path ./pre/model_session_0_cub.pth


python train.py --backbone_name resnet18 --model_name my --pretrained --dataset_name sketch --dataset_type cl_shot \
--optim sgd --opt opt2 --loss_type ce --no_order --seed 1993 --way 16 --shot 5 --session 17 --lr 0.1 --lr-scheduler cos \
--batch_size 128 --gpu-ids 2 --base_class 118 --val --data_dir ./datasets \
--model_path ./pre/model_session_0_sketch.pth
