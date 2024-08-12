import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk


def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        #target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

def calculate_dice_percase(pred, gt, num_classes):
    
    dice_total = 0

    for i in range(1, num_classes):
        pred_copy, gt_copy = pred.copy(), gt.copy()
        pred_copy[pred ==i] = 1
        gt_copy[gt == i] = 1
        pred_copy[pred != i] = 0
        gt_copy[gt != i] = 0
        if pred_copy.sum() > 0 and gt_copy.sum()>0:
            dice = metric.binary.dc(pred_copy, gt_copy)
            dice_total += dice
        elif pred.sum() > 0 and gt.sum()==0:
            dice_total += 1
    
    return dice_total

##### Adaptive tvMF Dice loss #####
class Adaptive_tvMF_DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(Adaptive_tvMF_DiceLoss, self).__init__()
        self.n_classes = n_classes

    ### one-hot encoding ###
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    ### tvmf dice loss ###
    def _tvmf_dice_loss(self, score, target, kappa):
        target = target.float()
        smooth = 1.0

        score = F.normalize(score, p=2, dim=[0,1,2])
        target = F.normalize(target, p=2, dim=[0,1,2])
        cosine = torch.sum(score * target)
        intersect =  (1. + cosine).div(1. + (1.- cosine).mul(kappa)) - 1.
        loss = (1 - intersect)**2.0

        return loss

    ### main ###
    def forward(self, inputs, target, kappa=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0.0
        for i in range(0, self.n_classes):
            tvmf_dice = self._tvmf_dice_loss(inputs[:, i], target[:, i], kappa[i])
            loss += tvmf_dice
        return loss / self.n_classes


class DiceScoreCoefficient(nn.Module):
    def __init__(self, n_classes):
        super(DiceScoreCoefficient, self).__init__()
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def fast_hist(self, label_true, label_pred, labels):
        mask = (label_true >= 0) & (label_true < labels)
        hist = np.bincount(labels * label_true[mask].astype(int) + label_pred[mask], minlength=labels ** 2,
        ).reshape(labels, labels)
        return hist

    def _dsc(self, mat):
        diag_all = np.sum(np.diag(mat))
        fp_all = mat.sum(axis=1)
        fn_all = mat.sum(axis=0)
        tp_tn = np.diag(mat)
        precision = np.zeros((self.n_classes)).astype(np.float32)
        recall = np.zeros((self.n_classes)).astype(np.float32)    
        f2 = np.zeros((self.n_classes)).astype(np.float32)

        for i in range(self.n_classes):
            if (fp_all[i] != 0)and(fn_all[i] != 0):   
                precision[i] = float(tp_tn[i]) / float(fp_all[i])
                recall[i] = float(tp_tn[i]) / float(fn_all[i])
                if (precision[i] != 0)and(recall[i] != 0):  
                     f2[i] = (2.0*precision[i]*recall[i]) / (precision[i]+recall[i])
                else:       
                    f2[i] = 0.0
            else:
                precision[i] = 0.0
                recall[i] = 0.0

        return f2

    ### main ###
    def forward(self, output, target):
        output = np.array(output)
        target = np.array(target)
        output = np.argmax(output,axis = 1)
        target = np.argmax(target, axis = 1)

        for lt, lp in zip(target, output):
            self.confusion_matrix += self.fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

        dsc = self._dsc(self.confusion_matrix)

        return dsc