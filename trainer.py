import argparse
from cgi import test
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, powerset, Adaptive_tvMF_DiceLoss, DiceScoreCoefficient
from torchvision import transforms
from utils import test_single_volume, calculate_metric_percase, calculate_dice_percase

from data.dataset import Huaxi_dataset, RandomGenerator

def trainer_huaxi(args, model):
    snapshot_path = args.output_dir
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    transform = RandomGenerator(output_size=[args.img_size, args.img_size],num_classes=args.num_classes)

    #transform = Huaxi_RandomGenerator(output_size=[args.img_size, args.img_size], args=args)
    db_train = Huaxi_dataset(args.task, 'train', args.base_dir, args.img_dir, args.label_dir, transform=transform)
    
    trainloader = DataLoader(db_train, batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True, drop_last=True)

    db_test = Huaxi_dataset(args.task, 'test', args.base_dir, args.img_dir, args.label_dir, transform=transform)
    testloader = DataLoader(db_test, batch_size=12,shuffle=True,num_workers=1,pin_memory=True)
    
    if args.n_gpu >1 :
        model = nn.DataParallel(model)
    
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = Adaptive_tvMF_DiceLoss(num_classes) if args.vmf_loss else DiceLoss(num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    best_dice = -1
    eval_iter = 0
    iterator = tqdm(range(max_epoch), ncols=70)
    l = [0, 1, 2, 3]
    ss = [x for x in powerset(l)]
    kappa = torch.Tensor(np.zeros((num_classes))).cuda()
    for epoch_num in iterator:
        for i_batch, (img, label) in enumerate(trainloader):
            image_batch, label_batch = img, label
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()


            # model.load_state_dict(torch.load(os.path.join(snapshot_path, "best_model.pth")))
            # torch.onnx.export(model, image_batch[0:1], 
            #                 os.path.join(snapshot_path, f"{args.task}.onnx"), 
            #                 export_params=True, 
            #                 input_names=['input'], 
            #                 output_names=['output'],
            #                 dynamic_axes= {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            #                 )
            # exit()
            outputs = model(image_batch)

            loss = 0.0
            lc1, lc2 = 0.3, 0.7
                    
            for s in ss:
                iout = 0.0
                if(s==[]):
                    continue
                #print(s)
                for idx in range(len(s)):
                    iout += outputs[s[idx]]
                loss_ce = ce_loss(iout, label_batch[:].float())
                loss_dice = dice_loss(iout, label_batch, kappa, softmax=True) if args.vmf_loss else dice_loss(iout, label_batch, softmax=True)
                loss += (lc1 * loss_ce + lc2 * loss_dice) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs[0], dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                labs = labs.squeeze(0)
                writer.add_image('train/GroundTruth', labs, iter_num)
        
        if args.vmf_loss:
            dsc = vmf_eval(model, testloader, args)
            kappa =  torch.Tensor(dsc * 32.0).cuda()
        #save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #torch.save(model.state_dict(), save_mode_path)

        save_interval = 50  #
        if (epoch_num + 1) >= 50 and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path + "/ckpts/", 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            eval_dice = evaluation(model, testloader, writer, args, eval_iter)
            eval_iter += 1
            if best_dice == -1:
                best_dice = eval_dice
            if eval_dice > best_dice:
                torch.save(model.state_dict(), os.path.join(snapshot_path, "best_model.pth"))
                best_dice = eval_dice

    save_mode_path = os.path.join(snapshot_path + "/ckpts/", 'epoch_' + str(epoch_num) + '.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    iterator.close()
    eval_dice = evaluation(model, testloader, writer, args, eval_iter)
    if eval_dice > best_dice:
        torch.save(model.state_dict(), os.path.join(snapshot_path, "best_model.pth"))
        best_dice = eval_dice
    
    model.load_state_dict(torch.load(os.path.join(snapshot_path, "best_model.pth")))
    torch.onnx.export(model, image_batch[0:1], 
                    os.path.join(snapshot_path, f"{args.task}.onnx"), 
                    export_params=True, 
                    input_names=['input'], 
                    output_names=['output'],
                    dynamic_axes= {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                    )

    writer.close()
    return "Training Finished!"



def evaluation(model, dataloader, writer, args, eval_iter):

    model.eval()
    metric_list = 0.0

    total_dice = 0
    with torch.no_grad():
        for i, (img,label) in enumerate(dataloader):
            img = img.cuda()
            label = label.cuda()
            outputs = model(img)
        
            out = torch.argmax(torch.softmax(outputs[0], dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()

            ###  visulization
            image = img[0, 0:1, :, :]
            image = (image - image.min()) / (image.max() - image.min())
            writer.add_image('test/Image', image, eval_iter * len(dataloader) + i)
            outputs = torch.argmax(torch.softmax(outputs[0], dim=1), dim=1, keepdim=True)
            writer.add_image('test/Prediction', outputs[0, ...] * 50, eval_iter * len(dataloader) + i)
            labs = label[0, ...].unsqueeze(0) * 50
            labs = labs.squeeze(0)
            writer.add_image('test/GroundTruth', labs, eval_iter * len(dataloader) + i)

            ###  metrics
            label = torch.argmax(label, dim=1).squeeze(0)
            label = label.cpu().detach().numpy()
            metric_i = []
            for j in range(args.num_classes):
                metric_i.append(calculate_metric_percase(out == j, label == j))
            
            metric_list += np.array(metric_i)

    metric_list = metric_list / len(dataloader)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]

    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    model.train()
    return performance


def vmf_eval(model, dataloader, args):
    model.eval()
    predict = []
    answer = []
    with torch.no_grad():
        for batch_idx, (img,label) in enumerate(dataloader):
            img = img.cuda()
            label = label.numpy()
            outputs = model(img)
            out = torch.softmax(outputs[0], dim=1)

            for i in range(len(label)):
                predict.append(out[i].cpu().numpy())
                answer.append(label[i])
        
        dsc = DiceScoreCoefficient(n_classes=args.num_classes)(predict, answer)

    model.train()
    return dsc