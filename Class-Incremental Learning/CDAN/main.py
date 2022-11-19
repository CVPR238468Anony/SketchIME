import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch.nn as nn
import torch
from model import *
from tqdm import tqdm
from sketchdataloader.sketch import SketchPngDataset
from torch.utils.data import DataLoader
from train import train
from test import test
import datetime
import numpy as np
from pprint import pprint

CUDA = True if torch.cuda.is_available() else False
LEARNING_RATE = 0.1
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

def main():
    now_time = datetime.datetime.now().__str__().replace(" ", "_")
    """
    This method puts all the modules together to train a neural network
    classifier using CORAL loss.

    Reference: https://arxiv.org/abs/1607.01719
    """
    parser = argparse.ArgumentParser(description="domain adaptation w CORAL")

    parser.add_argument("--epochs", default=50, type=int,help="number of training epochs")

    parser.add_argument("--batch_size_source", default=256, type=int,help="batch size of source data") 

    parser.add_argument("--batch_size_target", default=256, type=int,help="batch size of target data")
    
    parser.add_argument("--batch_size_test", default= 256, type=int)
    
    parser.add_argument("--num_session", default=17, type=int)
    
    parser.add_argument("--num_classes", default=374, type=int, help="no. classes in dataset (default 374)")
    
    parser.add_argument("--num_base", default=118, type=int, help="base class")
    
    parser.add_argument("--num_way", default=16, type=int, help="classes in each increments")
    
    parser.add_argument("--pretrain_dir", default=None, type=str, help="load pretrained model (default None)")
    
    parser.add_argument("--target_supervised", default=True, type=bool,help="using labels of target domain when cross-training")

    parser.add_argument("--result_path",default="experiment",type=str,help="save the result")

    parser.add_argument("--entropy", action="store_true", default=False, help="CDAN or CDAN+E")
    
    parser.add_argument("--dataset", default="inc", type=str)
    
    parser.add_argument("--dataroot", default="data", type=str)
    
    parser.add_argument("--num_workers", default=8, type=int)
    
    parser.add_argument("--lr", default=LEARNING_RATE, type=float)
    
    parser.add_argument("--ft", action="store_true", default=False, help="fine tune encoder")

    args = parser.parse_args()
    
    pprint(vars(args))
    
    # init dir
    work_path = os.path.dirname(__file__)
    datalist_dir = os.path.join(work_path, args.dataroot, "index_list", args.dataset)
    print("datalist position: {}".format(datalist_dir))
    
    result_path = os.path.join(work_path, args.result_path, now_time)
    if os.path.isdir(result_path) == False:
        os.makedirs(result_path)
    print("create result dir")
    result_file = open(os.path.join(result_path, "result.txt"), "w")
    
    test_acc_before_train = []
    test_acc_after_train = []
    # for each session
    for session in range(0, args.num_session):
        # class in this session
        class_set = args.num_base + session * args.num_way
        
        # get dataloader
        if session > 0:
            source_data = SketchPngDataset(root=args.dataroot, list_txt_url=os.path.join(datalist_dir, "source.txt"))
            source_loader = DataLoader(dataset=source_data, batch_size=args.batch_size_source, shuffle=True, num_workers=args.num_workers)
            target_data = SketchPngDataset(root=args.dataroot, list_txt_url=os.path.join(datalist_dir, "target_{}.txt").format(session))
            target_loader = DataLoader(dataset=target_data, batch_size=args.batch_size_target, shuffle=True, num_workers=args.num_workers)
        test_data = SketchPngDataset(root=args.dataroot, list_txt_url=os.path.join(datalist_dir, "test_{}.txt".format(session)))
        test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size_test, shuffle=False, num_workers=args.num_workers)
        
        # init network
        feature_dim = 512
        model = nn.DataParallel(baseNetwork(num_classes=class_set))
        ad_net = AdversarialNetwork(feature_dim * class_set, 1024)
        if CUDA:
            model = model.cuda()
            ad_net = ad_net.cuda()
        if args.pretrain_dir is not None:
            pretrain_model_path = os.path.join(args.pretrain_dir, "session{}_max_acc.pth".format(session))
            pretrain_params = torch.load(pretrain_model_path)["params"]
            
            # split out fc params.
            fc_params = pretrain_params["module.fc.weight"]
            del pretrain_params["module.fc.weight"]
            fc_params = fc_params[:class_set, :]
            
            model.load_state_dict(pretrain_params, False)
            model.module.fc.weight.data.copy_(fc_params)
            print("\ninit model param with {{{}}}".format(pretrain_model_path))
        
        # init optimizer
        if args.ft == True:
            # fine tune encoder.
            optimizer = torch.optim.SGD([
                {"params": model.module.encoder.parameters()},
                {"params": model.module.fc.parameters(), "lr":10*args.lr},
                {"params":ad_net.parameters(), "lr_mult": 10, 'decay_mult': 2}
            ], lr=args.lr, momentum=MOMENTUM)
        else:
            optimizer = torch.optim.SGD([
                {"params": model.module.fc.parameters(), "lr":10*args.lr},
                {"params":ad_net.parameters(), "lr_mult": 10, 'decay_mult': 2}
            ], lr=args.lr, momentum=MOMENTUM)
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.1)
        
        
        # before train
        print("-------------[Session{}, Before domain training]-------------".format(session))
        result_file.write("-------------[Session{}, Before domain training]-------------\n".format(session))
        print("session: {}, test label:\n".format(session), np.unique(test_loader.dataset.targets))
        test_tgttest = test(model, test_loader, -1, CUDA)
        print("session: {}, test_acc: {:.3f}".format(
            session,
            test_tgttest['accuracy %']
        ))
        result_file.write("session: {}, test_acc: {:.3f}\n".format(
            session,
            test_tgttest['accuracy %']
        ))
        test_acc_before_train.append(float("%.3f" % test_tgttest['accuracy %'].item()))
        
        if session == 0:
            test_acc_after_train.append(float("%.3f" % test_tgttest['accuracy %'].item()))
            continue
        
        # in train
        print("-------------[Session{}, Domain training]-------------".format(session))
        print("source label:\n", np.unique(source_loader.dataset.targets))
        print("target label:\n", np.unique(target_loader.dataset.targets))
        epoch_bar = tqdm(range(args.epochs))
        for epoch in epoch_bar:
            if args.ft == True:
                epoch_bar.set_description("session: {}, epoch: {}, lr_encoder: {:.4f}, lr_fc: {:.4f}, lr_ad: {:.4f}".format(
                    session, 
                    epoch, 
                    optimizer.param_groups[0]["lr"],
                    optimizer.param_groups[1]["lr"],
                    optimizer.param_groups[2]["lr"]
                ))
            else:
                epoch_bar.set_description("session: {}, epoch: {}, lr_fc: {:.4f}, lr_ad: {:.4f}".format(
                    session, 
                    epoch, 
                    optimizer.param_groups[0]["lr"],
                    optimizer.param_groups[1]["lr"],
                ))
            lambda_factor = (epoch + 1) / args.epochs
            model.train()
            ad_net.train()
            result_train = train(model, ad_net, source_loader, target_loader, optimizer, epoch, lambda_factor, CUDA, args.target_supervised, args.entropy)
            lr_schedule.step()
        
        # after train
        print("-------------[Session{}, After domain training]-------------".format(session))
        result_file.write("-------------[Session{}, After domain training]-------------\n".format(session))
        print("session: {}, test label:\n".format(session), np.unique(test_loader.dataset.targets))
        test_tgttest = test(model, test_loader, -1, CUDA)
        print("session: {}, test_acc: {:.3f}".format(
            session,
            test_tgttest['accuracy %']
        ))
        result_file.write("session: {}, test_acc: {:.3f}\n".format(
            session,
            test_tgttest['accuracy %']
        ))
        test_acc_after_train.append(float("%.3f" % test_tgttest['accuracy %'].item()))
        
        # save model's params
        save_path = os.path.join(result_path, "cdan-session{}.pth".format(session))
        torch.save(dict(params=model.state_dict()), save_path)
    
    result_file.write("Before train acc:\n")
    result_file.write("{}\n".format(test_acc_before_train))
    result_file.write("After train acc:\n")
    result_file.write("{}\n".format(test_acc_after_train))
    result_file.close()
    
if __name__ == "__main__":
    main()