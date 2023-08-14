from utils import *
import metrics
import metrics2
import fire
import numpy as np
import torch
import torchvision as tv
import torchvision
import os
import ipdb
import tqdm
import time
from calibration_methods.temperature_scaling import tune_temp
import calibration_methods.splines as splines
from calibration_methods.vector_scaling import VectorScaling, VectorScaling_NN

resnet50_files = [
                    # 'probs_resnet152_imgnet_logits.p',
                   'cal_resnet50_imgnet-v2-a_logits.p',
                   'cal_resnet50_imgnet-v2-b_logits.p',
                   'cal_resnet50_imgnet-v2-c_logits.p',
                   'cal_resnet50_imgnet-s_logits.p',
                   # 'cal_resnet152_imgnet-gauss_logits.p',
                   # 'cal_resnet152_imgnet-a_logits.p',
                   # 'cal_resnet152_imgnet-r_logits.p',
                   ]

resnet152_files = [
                    # 'probs_resnet152_imgnet_logits.p',
                   'cal_resnet152_imgnet-v2-a_logits.p',
                   'cal_resnet152_imgnet-v2-b_logits.p',
                   'cal_resnet152_imgnet-v2-c_logits.p',
                   'cal_resnet152_imgnet-s_logits.p',
                   'cal_resnet152_imgnet-a_logits.p',
                   'cal_resnet152_imgnet-r_logits.p',
                   ]

vit_small_patch32_224_files = [
                    'probs_vit_small_patch32_224_imgnet_logits.p',
                   'cal_vit_small_patch32_224_imgnet-v2-a_logits.p',
                   'cal_vit_small_patch32_224_imgnet-v2-b_logits.p',
                   'cal_vit_small_patch32_224_imgnet-v2-c_logits.p',
                   'cal_vit_small_patch32_224_imgnet-s_logits.p',
                   # 'cal_vit_small_patch32_224_imgnet-gauss_logits.p',
                   'cal_vit_small_patch32_224_imgnet-a_logits.p',
                   'cal_vit_small_patch32_224_imgnet-r_logits.p',
                   ]

deit_small_patch16_224_files = [
                    'probs_deit_small_patch16_224_imgnet_logits.p',
                   'cal_deit_small_patch16_224_imgnet-v2-a_logits.p',
                   'cal_deit_small_patch16_224_imgnet-v2-b_logits.p',
                   'cal_deit_small_patch16_224_imgnet-v2-c_logits.p',
                   'cal_deit_small_patch16_224_imgnet-s_logits.p',
                   # 'cal_deit_small_patch16_224_imgnet-gauss_logits.p',
                   'cal_deit_small_patch16_224_imgnet-a_logits.p',
                   'cal_deit_small_patch16_224_imgnet-r_logits.p',
                   ]

deit_base_patch16_224_files = [
                    'probs_deit_base_patch16_224_imgnet_logits.p',
                   'cal_deit_base_patch16_224_imgnet-v2-a_logits.p',
                   'cal_deit_base_patch16_224_imgnet-s_logits.p',
                   'cal_deit_base_patch16_224_imgnet-gauss_logits.p',
                   'cal_deit_base_patch16_224_imgnet-a_logits.p',
                   ]
dct = {}
dct['resnet50_files'] = resnet50_files
dct['resnet152_files'] = resnet152_files
dct['vit_small_patch32_224_files'] = vit_small_patch32_224_files
dct['deit_small_patch16_224_files'] = deit_small_patch16_224_files
dct['deit_base_patch16_224_files'] = deit_base_patch16_224_files

def demo(files='resnet152_files'):
    for f in dct[files]:
        print('###############################################################################')
        print('Evaluating : {}'.format(f))
        dataset = f.split('_')[-2]
        if dataset == 'imgnet-a':
            valid_indices = imagenet_a_valid_labels()
        elif dataset == 'imgnet-r':
            valid_indices = imagenet_r_valid_labels()
        else:
            valid_indices = None
        # set your path of logits here
        path = os.path.join('./logits',f)
        (y_probs_val, y_val), (y_probs_test, y_test) = unpickle_probs(path)
        y_val = np.squeeze(y_val)
        y_test = np.squeeze(y_test)

        avg_conf_val, _ = metrics.AvgConf(torch.from_numpy(y_probs_val).cuda(), torch.from_numpy(y_val).cuda())
        if valid_indices:
            avg_conf_test, _ = metrics.AvgConf(torch.from_numpy(y_probs_test[:, valid_indices]).cuda(), torch.from_numpy(y_test).cuda())
        else:
            avg_conf_test, _ = metrics.AvgConf(torch.from_numpy(y_probs_test).cuda(), torch.from_numpy(y_test).cuda())
        alpha = avg_conf_test/avg_conf_val


        # r = 0
        r1 = 0
        print('n_T/n_P : {}'.format(r1))
        sampled_logits, sampled_labels = sample_calibration_set(torch.from_numpy(y_probs_val), torch.from_numpy(y_val), r1)

        temp = tune_temp(torch.from_numpy(sampled_logits), torch.from_numpy(np.squeeze(sampled_labels)))
        if valid_indices:
            logits_temp_0 = y_probs_test[:, valid_indices] / temp
        else:
            logits_temp_0 = y_probs_test / temp

        ece_criterion = splines._ECELoss(n_bins=25)
        softmax_val = torch.nn.functional.softmax(torch.from_numpy(sampled_logits), dim=1).cpu().numpy()
        if valid_indices:
            softmax_test = torch.nn.functional.softmax(torch.from_numpy(y_probs_test[:, valid_indices]), dim=1).cpu().numpy()
        else:
            softmax_test = torch.nn.functional.softmax(torch.from_numpy(y_probs_test), dim=1).cpu().numpy()
        confs_spline_0, labels_spline_0 = splines.cal_splines(softmax_val, sampled_labels, softmax_test, y_test, ece_criterion)

        model = VectorScaling_NN(classes=y_probs_val.shape[1]).cuda()
        torch_sample_logits = torch.from_numpy(sampled_logits).cuda()
        torch_sample_labels = torch.from_numpy(sampled_labels).cuda()
        model.fit(torch_sample_logits, torch_sample_labels)
        torch_test_logits = torch.from_numpy(y_probs_test).cuda()
        # torch_test_labels = torch.from_numpy(np.squeeze(y_test)).cuda()
        preds_test_0 = model(torch_test_logits).detach()
        if valid_indices:
            preds_test_0 = preds_test_0[:, valid_indices]

        # r = 2
        r2 = 0.1
        print('n_T/n_P : {}'.format(r2))
        sampled_logits, sampled_labels = sample_calibration_set(torch.from_numpy(y_probs_val), torch.from_numpy(y_val), r2)
        avg_conf_cal, _ = metrics.AvgConf(torch.from_numpy(sampled_logits).cuda(), torch.from_numpy(sampled_labels).cuda())
        # alpha = 1-abs((avg_conf_test-avg_conf_val)/(avg_conf_cal-avg_conf_val))

        temp = tune_temp(torch.from_numpy(sampled_logits), torch.from_numpy(np.squeeze(sampled_labels)))
        if valid_indices:
            logits_temp_1 = y_probs_test[:, valid_indices] / temp
        else:
            logits_temp_1 = y_probs_test / temp

        ece_criterion = splines._ECELoss(n_bins=25)
        softmax_val = torch.nn.functional.softmax(torch.from_numpy(sampled_logits), dim=1).cpu().numpy()
        if valid_indices:
            softmax_test = torch.nn.functional.softmax(torch.from_numpy(y_probs_test[:, valid_indices]),
                                                       dim=1).cpu().numpy()
        else:
            softmax_test = torch.nn.functional.softmax(torch.from_numpy(y_probs_test), dim=1).cpu().numpy()
        confs_spline_1, labels_spline_1 = splines.cal_splines(softmax_val, sampled_labels, softmax_test, y_test, ece_criterion)

        model = VectorScaling_NN(classes=y_probs_val.shape[1]).cuda()
        torch_sample_logits = torch.from_numpy(sampled_logits).cuda()
        torch_sample_labels = torch.from_numpy(sampled_labels).cuda()
        model.fit(torch_sample_logits, torch_sample_labels)
        torch_test_logits = torch.from_numpy(y_probs_test).cuda()
        # torch_test_labels = torch.from_numpy(np.squeeze(y_test)).cuda()
        preds_test_1 = model(torch_test_logits).detach()
        if valid_indices:
            preds_test_1 = preds_test_1[:, valid_indices]

        # ensemble
        logits_temp = alpha*logits_temp_0 + (1-alpha)*logits_temp_1
        torch_test_logits = torch.from_numpy(logits_temp).cuda()
        torch_test_labels = torch.from_numpy(y_test).cuda()
        avg_conf, acc = metrics.AvgConf(torch_test_logits, torch_test_labels)
        acc, ece = metrics.ECE(torch_test_logits, torch_test_labels)
        # acc, mce = metrics.MCE(torch_test_logits, torch_test_labels, isLogits=0)
        # nll = metrics.NLL(torch_test_logits, torch_test_labels, 0)
        # bs = metrics.BS(torch_test_logits, torch_test_labels, 0)
        # softmax_test = torch.nn.functional.softmax(torch.from_numpy(logits_temp), dim=1).cpu().numpy()
        # cw_ece = metrics2.classwise_ECE(softmax_test, y_test)
        print("Avg Conf Val : %4f, Cal : %4f, Test : %4f" %(avg_conf_val, avg_conf_cal, avg_conf_test))
        print('temp logits ensemble -------> ')
        print('ACC: %4f' % (acc))
        # print('NLL: %4f' % (nll))
        # print('BS: %4f' % (bs))
        # print('MCE: %4f' % (mce))
        print('AVG Conf: %4f' % (avg_conf))
        print('conf-ECE: %4f' % (ece))
        # print('cw-ECE: %4f' % (cw_ece))

        softmax_0 = torch.nn.functional.softmax(torch.from_numpy(logits_temp_0), dim=1).cuda()
        softmax_1 = torch.nn.functional.softmax(torch.from_numpy(logits_temp_1), dim=1).cuda()
        softmax_temp = alpha*softmax_0+softmax_1*(1-alpha)
        avg_conf, acc = metrics.AvgConf(softmax_temp, torch_test_labels, isLogits=1)
        acc, ece = metrics.ECE(softmax_temp, torch_test_labels, isLogits=1)
        print('temp softmax ensemble -------> ')
        print('AVG Conf: %4f' % (avg_conf))
        print('conf-ECE: %4f' % (ece))

        confs_spline = alpha*confs_spline_0 + confs_spline_1*(1-alpha)
        avg_conf, acc = metrics.AvgConf(torch.from_numpy(confs_spline), torch.from_numpy(np.squeeze(labels_spline_1)), isLogits=2)
        acc, ece = metrics.ECE(torch.from_numpy(confs_spline), torch.from_numpy(np.squeeze(labels_spline_1)), isLogits=2)
        print('splines (confidence) ensemble -------> ')
        print('ACC: %4f' % (acc))
        print('AVG Conf: %4f' % (avg_conf))
        print('conf-ECE: %4f' % (ece))

        softmax_0 = torch.nn.functional.softmax(preds_test_0, dim=1)
        softmax_1 = torch.nn.functional.softmax(preds_test_1, dim=1)
        softmax_temp = alpha * softmax_0 + softmax_1 * (1 - alpha)
        avg_conf, acc = metrics.AvgConf(softmax_temp, torch_test_labels, isLogits=1)
        acc, ece = metrics.ECE(softmax_temp, torch_test_labels, isLogits=1)
        print('vector scaling softmax ensemble -------> ')
        print('ACC: %4f' % (acc))
        print('AVG Conf: %4f' % (avg_conf))
        print('conf-ECE: %4f' % (ece))

if __name__ == '__main__':
    fire.Fire(demo)
