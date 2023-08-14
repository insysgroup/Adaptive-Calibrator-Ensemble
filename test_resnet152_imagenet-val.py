import numpy as np

import torch
import torchvision as tv
import torchvision
import os
import ipdb
import time
import logging
import tqdm

from utils import *
from sklearn.model_selection import train_test_split

# import timm
# from timm.data import resolve_data_config
# from timm.data.transforms_factory import create_transform

if __name__ == '__main__':
    model = tv.models.resnet152(pretrained=True).cuda()

    mean = [0.485, 0.456, 0.406]
    stdv = [0.229, 0.224, 0.225]
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])
    # # -------------------------- dataloader ----------------------------
    # root = '../data/imagenet-v2-a'
    # dataset_test = tv.datasets.ImageFolder(root, transforms)
    # loader_test = torch.utils.data.DataLoader(dataset_test, pin_memory=True, batch_size=32, num_workers=18)
    # -------------------------- dataloader ----------------------------
    dataset = tv.datasets.ImageFolder('imagenet-val', transforms)
    # image names
    data_info = np.array(dataset.imgs)[:, 0]

    list_names = []
    for i in range(data_info.shape[0]):
        item = data_info[i]
        list_names.append(item.split('/')[-1])

    # sort indice
    indice = np.argsort(list_names)

    # sorted imgs
    imgs = np.array(dataset.imgs)[indice]

    # select data for val (carlibration) and test
    from sklearn.model_selection import train_test_split

    x_test, x_val, _, _ = train_test_split(imgs, imgs, test_size=0.5, random_state=333)

    # change the label type
    x_test = x_test.tolist()
    x_val = x_val.tolist()

    for i in range(len(x_val)):
        x_val[i][1] = int(x_val[i][1])

    for i in range(len(x_test)):
        x_test[i][1] = int(x_test[i][1])

    # ===================================================== #
    # ---------    data loaders for calibration   --------- #
    # ===================================================== #
    dataset_val = tv.datasets.ImageFolder('imagenet-val',
                                          transforms)
    dataset_val.samples = x_val
    dataset_val.imgs = x_val

    dataset_test = tv.datasets.ImageFolder('imagenet-val',
                                           transforms)
    dataset_test.samples = x_test
    dataset_test.imgs = x_test

    loader_val = torch.utils.data.DataLoader(dataset_val, pin_memory=True, batch_size=32, num_workers=32)
    loader_test = torch.utils.data.DataLoader(dataset_test, pin_memory=True, batch_size=32, num_workers=32)

    val_logits_list = []
    val_labels_list = []
    model.eval()
    for input, label in tqdm.tqdm(loader_val):
        with torch.no_grad():
            input = input.cuda()
            label = label.cuda()
            logit, _ = model(input)

        val_logits_list.append(logit)
        val_labels_list.append(label)
    val_logits = torch.cat(val_logits_list).cuda().cpu().numpy()
    val_labels = torch.cat(val_labels_list).cuda().cpu().numpy()

    softmaxes =  torch.nn.functional.softmax(torch.from_numpy(val_logits), dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(torch.from_numpy(val_labels))
    acc = accuracies.float().sum() / len(accuracies)
    print("Acc : %f" %(acc))
    
    test_logits_list = []
    test_labels_list = []
    for input, label in tqdm.tqdm(loader_test):
        with torch.no_grad():
            input = input.cuda()
            label = label.cuda()
            logit, _ = model(input)

        test_logits_list.append(logit)
        test_labels_list.append(label)
    test_logits = torch.cat(test_logits_list).cuda().cpu().numpy()
    test_labels = torch.cat(test_labels_list).cuda().cpu().numpy()

    softmaxes =  torch.nn.functional.softmax(torch.from_numpy(test_logits), dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(torch.from_numpy(test_labels))
    acc = accuracies.float().sum() / len(accuracies)
    print("Acc : %f" %(acc))
    import pickle
    with open('logits/probs_resnet152_imgnet_logits.p', 'wb') as f:
        pickle.dump([(val_logits, val_labels), (test_logits, test_labels)], f)
