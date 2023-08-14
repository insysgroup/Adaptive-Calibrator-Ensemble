import numpy as np
import os
import pickle
from utils import *

models = ['deit_small_patch16_224', 'vit_small_patch32_224','resnet152'] #'deit_base_patch16_224',
datasets = ['imgnet-v2-b', 'imgnet-v2-c'] #'imgnet-v2-a', 'imgnet-s',

for m in  models:
    path1 = os.path.join('../logits', 'probs_{}_imgnet_logits.p'.format(m))
    (y_probs_val, y_val), (_, _) = unpickle_probs(path1, True)
    for d in datasets:
        if d == 'imgnet-gauss':
            path2 = os.path.join('../logits', 'probs_{}_{}_logits.p'.format(m, d))
            (_, _), (y_probs_test, y_test) = unpickle_probs(path2, True)
        else:
            path2 = os.path.join('../logits', 'probs_{}_{}_logits.p'.format(m, d))
            with open(path2, 'rb') as f:
                y_probs_test, y_test = pickle.load(f)


        with open(os.path.join('../logits', 'cal_{}_{}_logits.p'.format(m, d)), 'wb') as f:
            pickle.dump([(y_probs_val, y_val), (y_probs_test, y_test)], f)