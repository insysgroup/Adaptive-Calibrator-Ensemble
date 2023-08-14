# OOD-calibration

This is the implement of paper:
___Adaptive Calibrator Ensemble for Predictive Calibration Under Distribution Shift___


### Dataset

We list the links of the used datasets and check carefully their licenses for our usage: 
[ImageNet-Validation](https://www.image-net.org),
[ImageNet-V2](https://github.com/modestyachts/ImageNetV2),
[ImageNet-C(orruption)](https://github.com/hendrycks/robustness),
[ImageNet-S(ketch)](https://github.com/HaohanWang/ImageNet-Sketch),
[ImageNet-Adv(ersarial)](https://github.com/hendrycks/natural-adv-examples),
[ImageNet-R(endition)](https://github.com/hendrycks/imagenet-r),
[CIFAR-10](https://www.cs.toronto.edu/kriz/cifar.html),
[CIFAR-10-C](https://github.com/hendrycks/robustness).

### Preparation
Logits are generated with "test_(model)_(dataset.py)" and "gen_ood_logits.py". 
Logits are stored in `logits` dir.
```
resnet152_files = [
                    # 'probs_resnet152_imgnet_logits.p',
                   'cal_resnet152_imgnet-v2-a_logits.p',
                   'cal_resnet152_imgnet-v2-b_logits.p',
                   'cal_resnet152_imgnet-v2-c_logits.p',
                   'cal_resnet152_imgnet-s_logits.p',
                   'cal_resnet152_imgnet-a_logits.p',
                   'cal_resnet152_imgnet-r_logits.p',
                   ]
```

```
(y_probs_val, y_val), (y_probs_test, y_test) = unpickle_probs(path)
```

### Usage

Run our ensemble method with these three baselines (*i.e.,* Vector Scaling, Temperature Scaling, Spline):
```
python run_ensemble.py --files <'resnet50_files', 'resnet152_files', 'vit_small_patch32_224_files', 'deit_small_patch16_224_files'>
```

Run our alternative method with these three baselines (*i.e.,* Vector Scaling, Temperature Scaling, Spline):
```
python run_estimate_ratio.py --files <'resnet50_files', 'resnet152_files', 'vit_small_patch32_224_files', 'deit_small_patch16_224_files'>
```

