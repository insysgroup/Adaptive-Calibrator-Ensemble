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
from calibration_methods.histogram_binning import HistogramBinning
import calibration_methods.splines as splines
from calibration_methods.vector_scaling import VectorScaling, VectorScaling_NN


resnet152_files = [
                    # 'probs_resnet152_imgnet_logits.p',
                   'cal_resnet152_imgnet-v2-a_logits.p',
                   'cal_resnet152_imgnet-v2-b_logits.p',
                   'cal_resnet152_imgnet-v2-c_logits.p',
                   'cal_resnet152_imgnet-s_logits.p',
                   # 'cal_resnet152_imgnet-gauss_logits.p',
                   'cal_resnet152_imgnet-a_logits.p',
                   'cal_resnet152_imgnet-r_logits.p',
                   ]

vit_small_patch32_224_files = [
                    # 'probs_vit_small_patch32_224_imgnet_logits.p',
                   'cal_vit_small_patch32_224_imgnet-v2-a_logits.p',
                   'cal_vit_small_patch32_224_imgnet-v2-b_logits.p',
                   'cal_vit_small_patch32_224_imgnet-v2-c_logits.p',
                   'cal_vit_small_patch32_224_imgnet-s_logits.p',
                   # 'cal_vit_small_patch32_224_imgnet-gauss_logits.p',
                   'cal_vit_small_patch32_224_imgnet-a_logits.p',
                   'cal_vit_small_patch32_224_imgnet-r_logits.p',
                   ]

deit_small_patch16_224_files = [
                    # 'probs_deit_small_patch16_224_imgnet_logits.p',
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
imagenetwild_files = [
                    'cal_resnet50-0_img-wild-id_logits.p',
                   'cal_resnet50-0_img-wild-ood_logits.p',
                   'cal_resnet50-1_img-wild-id_logits.p',
                   'cal_resnet50-1_img-wild-ood_logits.p',
                   'cal_resnet50-2_img-wild-id_logits.p',
                   'cal_resnet50-2_img-wild-ood_logits.p',

                   ]

dct = {}
dct['resnet152_files'] = resnet152_files
dct['vit_small_patch32_224_files'] = vit_small_patch32_224_files
dct['deit_small_patch16_224_files'] = deit_small_patch16_224_files
dct['deit_base_patch16_224_files'] = deit_base_patch16_224_files

dct['imagenetwild_files'] = imagenetwild_files

def ThresConf(logits, labels):
    softmaxes = torch.nn.functional.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    # acc = accuracies.float().sum() / len(accuracies)
    num = int(accuracies.float().sum())
    confidences = confidences.cpu().numpy()
    confidences = np.sort(confidences)
    pred_conf = confidences[-num]
    return pred_conf

def PredAcc(logits, threshold):
    softmaxes = torch.nn.functional.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    confidences = confidences.cpu().numpy()
    confidences = np.sort(confidences)
    confidences[confidences >= threshold] = 1
    confidences[confidences < threshold] = 0
    return confidences.mean()

def demo(files='resnet152_files'):
    for f in dct[files]:
        dataset = f.split('_')[-2]
        if dataset == 'imgnet-a':
            valid_indices = imagenet_a_valid_labels()
        elif dataset == 'imgnet-r':
            valid_indices = imagenet_r_valid_labels()
        else:
            valid_indices = None
        print('###############################################################################')
        print('Evaluating : {}'.format(f))
        path = os.path.join('./logits',f)
        (y_probs_val, y_val), (y_probs_test, y_test) = unpickle_probs(path)
        y_val = np.squeeze(y_val)
        y_test = np.squeeze(y_test)

        avg_conf_val, acc_val = metrics.AvgConf(torch.from_numpy(y_probs_val).cuda(), torch.from_numpy(y_val).cuda())
        avg_conf_test, _ = metrics.AvgConf(torch.from_numpy(y_probs_test).cuda(), torch.from_numpy(y_test).cuda())

        # uncalibrated
        print("Calibration Method : Uncalibrated")
        torch_val_logits = torch.from_numpy(y_probs_val).cuda()
        torch_val_labels = torch.from_numpy(y_val).cuda()
        # acc, ece = metrics.ECE(torch_val_logits, torch_val_labels)
        avg_conf, acc = metrics.AvgConf(torch_val_logits, torch_val_labels)
        print('val acc: %4f ' %(acc))
        print('val average confidence: %4f' %(avg_conf))
        if valid_indices:
            torch_test_logits = torch.from_numpy(y_probs_test[:, valid_indices]).cuda()
        else:
            torch_test_logits = torch.from_numpy(y_probs_test).cuda()
        torch_test_labels = torch.from_numpy(y_test).cuda()
        avg_conf, acc = metrics.AvgConf(torch_test_logits, torch_test_labels)
        acc, ece = metrics.ECE(torch_test_logits, torch_test_labels)
        acc, mce = metrics.MCE(torch_test_logits, torch_test_labels, isLogits=0)
        nll = metrics.NLL(torch_test_logits, torch_test_labels, 0)
        bs = metrics.BS(torch_test_logits, torch_test_labels, 0)
        softmax_test = torch.nn.functional.softmax(torch.from_numpy(y_probs_test), dim=1).cpu().numpy()
        cw_ece = metrics2.classwise_ECE(softmax_test, y_test)
        print('ACC: %4f' % (acc))
        print('NLL: %4f' % (nll))
        print('BS: %4f' % (bs))
        print('MCE: %4f' % (mce))
        print('AVG Conf: %4f' %(avg_conf))
        print('conf-ECE: %4f' % (ece))
        print('cw-ECE: %4f' % (cw_ece))

        est_error = (1-acc_val) + 2*(avg_conf_val- avg_conf_test)
        assert est_error>0
        est_acc = 1 - est_error
        r = est_acc/(1-est_acc)
        # thres_conf = ThresConf(torch_val_logits, torch_val_labels)
        # pred_acc = PredAcc(torch_test_logits, thres_conf)
        # r = pred_acc/(1-pred_acc)

        # ratios = [0, 0.5, 1.0, 2.0]
        # for r in ratios:
        print("===================================================================")
        print("sample ratio (n_T/n_P) : {}".format(r))
        # sample calibration set
        sampled_logits, sampled_labels = sample_calibration_set(torch.from_numpy(y_probs_val), torch.from_numpy(y_val), r)

        # temperature scaling
        print("Calibration Method : Temperature Scaling")
        temp = tune_temp(torch.from_numpy(sampled_logits), torch.from_numpy(np.squeeze(sampled_labels)))
        if valid_indices:
            logits_temp = y_probs_test[:, valid_indices] / temp
        else:
            logits_temp = y_probs_test / temp
        torch_test_logits = torch.from_numpy(logits_temp).cuda()
        torch_test_labels = torch.from_numpy(y_test).cuda()
        avg_conf, acc = metrics.AvgConf(torch_test_logits, torch_test_labels)
        acc, ece = metrics.ECE(torch_test_logits, torch_test_labels)
        acc, mce = metrics.MCE(torch_test_logits, torch_test_labels, isLogits=0)
        nll = metrics.NLL(torch_test_logits, torch_test_labels, 0)
        bs = metrics.BS(torch_test_logits, torch_test_labels, 0)
        softmax_test = torch.nn.functional.softmax(torch.from_numpy(logits_temp), dim=1).cpu().numpy()
        cw_ece = metrics2.classwise_ECE(softmax_test, y_test)
        print('ACC: %4f' % (acc))
        print('NLL: %4f' % (nll))
        print('BS: %4f' % (bs))
        print('MCE: %4f' % (mce))
        print('AVG Conf: %4f' % (avg_conf))
        print('conf-ECE: %4f' % (ece))
        print('cw-ECE: %4f' % (cw_ece))

        # spline
        print("Calibration Method : Spline")
        # np.random.seed(100)
        ece_criterion = splines._ECELoss(n_bins=25)
        softmax_val = torch.nn.functional.softmax(torch.from_numpy(sampled_logits), dim=1).cpu().numpy()
        if valid_indices:
            softmax_test = torch.nn.functional.softmax(torch.from_numpy(y_probs_test[:, valid_indices]), dim=1).cpu().numpy()
        else:
            softmax_test = torch.nn.functional.softmax(torch.from_numpy(y_probs_test), dim=1).cpu().numpy()
        confidences_test, labels_test = splines.cal_splines(softmax_val, sampled_labels, softmax_test, y_test, ece_criterion)

        avg_conf, acc = metrics.AvgConf(torch.from_numpy(confidences_test).cuda(),
                                        torch.from_numpy(labels_test).cuda(), isLogits=2)
        print('AVG Conf: %4f' % (avg_conf))
        # continue
        # vector scaling
        """
        print("Calibration Method : Vector Scaling")
        model = VectorScaling_NN(classes=y_probs_val.shape[1]).cuda()
        torch_sample_logits = torch.from_numpy(sampled_logits).cuda()
        torch_sample_labels = torch.from_numpy(sampled_labels).cuda()
        model.fit(torch_sample_logits, torch_sample_labels)
        torch_test_logits = torch.from_numpy(y_probs_test).cuda()
        torch_test_labels = torch.from_numpy(np.squeeze(y_test)).cuda()
        preds_test = model(torch_test_logits).detach()
        if valid_indices:
            preds_test = preds_test[:, valid_indices]
        acc, ece = metrics.ECE(preds_test, torch_test_labels, isLogits=0)
        avg_conf, acc = metrics.AvgConf(preds_test, torch_test_labels, isLogits=0)
        print('ACC: %4f' % (acc))
        print('AVG Conf: %4f' % (avg_conf))
        print('conf-ECE: %4f' % (ece))
        """
        # Matrix Scaling
        """
        # histogram binning
        print("Calibration Method : Histogram Binning")

        val_softmax = torch.nn.functional.softmax(torch.from_numpy(sampled_logits).cuda(), dim=1).cpu().numpy()
        test_softmax = torch.nn.functional.softmax(torch.from_numpy(y_probs_test).cuda(), dim=1).cpu().numpy()
        probs_test = np.zeros(test_softmax.shape)
        K = test_softmax.shape[1]
        sampled_labels = sampled_labels.reshape(-1,1)
        for k in tqdm.tqdm(range(K)):
            y_cal = np.array(sampled_labels == k, dtype="int")[:, 0]
            model = HistogramBinning()
            model.fit(val_softmax[:, k], y_cal)
            probs_test[:, k] = model.predict(test_softmax[:, k])
            idx_nan = np.where(np.isnan(probs_test))
            probs_test[idx_nan] = 0

        torch_logits = torch.from_numpy(probs_test).cuda()
        torch_labels = torch.from_numpy(np.squeeze(y_test)).cuda()

        acc, ece = metrics.ECE(torch_logits, torch_labels, isLogits=1)
        avg_conf, acc = metrics.AvgConf(torch_logits, torch_labels, isLogits=1)
        # nll = NLL(torch_logits, torch_labels)
        # bs = BS(torch_logits, torch_labels)
        # logits = torch.nn.functional.softmax(torch_logits, dim=1).cpu().numpy()
        # conf_ece = guo_ECE(probs_test, y_test)
        # cw_ece = classwise_ECE(probs_test, y_test)

        print('ACC: %4f' % (acc))
        # print('NLL: %4f' % (nll))
        # print('BS: %4f' % (bs))
        # print('MCE: %4f' % (mce))
        print('AVG Conf: %4f' % (avg_conf))
        print('conf-ECE: %4f' % (ece))
        # print('cw-ECE: %4f' % (cw_ece))
        """
        # dirichlet

if __name__ == '__main__':
    fire.Fire(demo)
