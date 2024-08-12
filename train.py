import argparse
import logging
import os
import os.path as osp
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from lib.networks import EMCADNet
from trainer import trainer_huaxi

parser = argparse.ArgumentParser()

parser.add_argument('--task_name', type=str,
                    default='ca', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--output_dir', default ='data/huaxiproj/EMCAD',type=str, help='output dir')
parser.add_argument('--data_dir', default ='data/huaxiproj',type=str, help='output dir')                    
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=500, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--vmf_loss', action="store_true")

args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # args.vmf_loss = True

    args.task = args.task_name
    args.num_classes = args.num_classes
    args.output_dir = osp.join(args.output_dir, args.task)
    if args.vmf_loss:
        args.output_dir += "/vmf"
    args.base_dir = osp.join(args.data_dir, args.task) if args.task not in ['bvr', 'ca'] \
        else osp.join(args.data_dir, "bvr_ca")
    args.img_dir = osp.join(args.data_dir, args.task, "images") if args.task not in ['bvr', 'ca'] \
        else osp.join(args.data_dir, "bvr_ca", "images")
    args.label_dir = osp.join(args.data_dir, args.task, "labels") if args.task not in ['bvr', 'ca'] \
        else osp.join(args.data_dir, "bvr_ca", "labels")
    args.is_pretrain = True

    if args.batch_size != 12 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 12

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(osp.join(args.output_dir, "ckpts"))
    net = EMCADNet(num_classes=args.num_classes).cuda()
    
    trainer_huaxi(args, net)