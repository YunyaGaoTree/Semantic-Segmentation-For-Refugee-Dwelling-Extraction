import os
from typing import Callable, List, Tuple
import argparse
import pathlib
from pathlib import Path 
import random
import numpy as np
from sklearn.model_selection import train_test_split
import collections

from skimage.io import imread as gif_imread
import imageio.v2 as imageio
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torch import nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch import optim

import albumentations as albu
from albumentations.pytorch import ToTensor

import catalyst
from catalyst import utils
from catalyst.contrib.nn import DiceLoss, IoULoss
from catalyst.contrib.nn import RAdam, Lookahead
from catalyst.dl import DiceCallback, IouCallback, CriterionCallback, MetricAggregationCallback
from catalyst.contrib.callbacks import DrawMasksCallback
from catalyst.dl import SupervisedRunner


# from python 3.10, some modules in utils change, so add this part
import sys

if sys.version_info.major == 3 and sys.version_info.minor >= 10:
    import collections
    setattr(collections, "MutableMapping", collections.abc.MutableMapping)
    
from catalyst import utils

print(f"torch: {torch.__version__}, catalyst: {catalyst.__version__}")

SEED = 42
utils.set_global_seed(SEED)
utils.prepare_cudnn(deterministic=True)

def show_examples(name: str, image: np.ndarray, mask: np.ndarray):
    plt.figure(figsize=(10, 14))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Image: {name}")

    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title(f"Mask: {name}")


def show(index: int, images: List[Path], masks: List[Path], transforms=None) -> None:
    image_path = images[index]
    mask_path = masks[index]
    name = image_path.name

    image = imageio.imread(image_path)
    mask = imageio.imread(mask_path)

    if transforms is not None:
        temp = transforms(image=image, mask=mask)
        image = temp["image"]
        mask = temp["mask"]

    show_examples(name, image, mask)

def show_random(images: List[Path], masks: List[Path], transforms=None) -> None:
    length = len(images)
    index = random.randint(0, length - 1)
    show(index, images, masks, transforms)
    
    
class SegmentationDataset(Dataset):
    def __init__(
        self,
        images: List[Path],
        masks: List[Path] = None,
        transforms=None
    ) -> None:
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.images[idx]
        image = utils.imread(image_path)
        
        result = {"image": image}
        
        if self.masks is not None:
            mask = gif_imread(self.masks[idx])
            mask = mask[...,0]
            # mask[mask > 0] = 1
            result["mask"] = mask
        
        if self.transforms is not None:
            result = self.transforms(**result)
        
        result["filename"] = image_path.name

        return result
    

def pre_transforms(image_size=224):
    return [albu.Resize(image_size, image_size, p=1)]


def hard_transforms():
    result = [
      albu.RandomRotate90(p=1),
      albu.HorizontalFlip(p=1),
      albu.VerticalFlip(p=1),
      albu.RandomBrightnessContrast(
          brightness_limit=0.2, contrast_limit=0.2, p=0.5
      ),
      albu.GridDistortion(p=0.5),
      albu.HueSaturationValue(p=0.5)
    ]

    return result


# to make the data size similar, I remove some of the data augmentation techniques here
def hard_transforms_SR():
    result = [
      albu.RandomRotate90(p=1),
      albu.HorizontalFlip(p=1),
      # albu.VerticalFlip(p=1),
      albu.RandomBrightnessContrast(
          brightness_limit=0.2, contrast_limit=0.2, p=0.5
      ),
      # albu.GridDistortion(p=0.5),
      # albu.HueSaturationValue(p=0.5)
    ]

    return result
  


def resize_transforms(image_size=224):
    BORDER_CONSTANT = 0
    # pre_size = int(image_size * 1.5) # this code is wrong, pre_size should be 1024
    pre_size = 1024

    random_crop = albu.Compose([
      albu.SmallestMaxSize(pre_size, p=1),
      albu.RandomCrop(
          image_size, image_size, p=1
      )
    ])

    # rescale = albu.Compose([albu.Resize(image_size, image_size, p=1)])

    random_crop_big = albu.Compose([
      albu.LongestMaxSize(pre_size, p=1),
      albu.RandomCrop(
          image_size, image_size, p=1
      )
    ])

    # Converts the image to a square of size image_size x image_size
    result = [
      albu.OneOf([
          random_crop,
          # rescale,
          random_crop_big
      ], p=1)
    ]

    return result
  
def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [albu.Normalize(), ToTensor()]
  
def compose(transforms_to_compose):
    # combine all augmentations into single pipeline
    result = albu.Compose([
      item for sublist in transforms_to_compose for item in sublist
    ])
    return result


def get_loaders(
    images_tr: List[Path],
    masks_tr: List[Path],
    images_val: List[Path],
    masks_val: List[Path],
    random_state: int,
    valid_size: float = 0.1,
    batch_size: int = 8,
    num_workers: int = 4,
    train_transforms_fn = None,
    valid_transforms_fn = None,
) -> dict:

    np_images_tr = np.array(images_tr)
    np_masks_tr = np.array(masks_tr)
    
    np_images_val = np.array(images_val)
    np_masks_val = np.array(masks_val)

    train_dataset = SegmentationDataset(
      images = np_images_tr.tolist(),
      masks = np_masks_tr.tolist(),
      transforms = train_transforms_fn
    )

    valid_dataset = SegmentationDataset(
      images = np_images_val.tolist(),
      masks = np_masks_val.tolist(),
      transforms = valid_transforms_fn
    )

    # Catalyst uses normal torch.data.DataLoader
    train_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      drop_last=True,
    )
    
    batch_size_val = 1
    valid_loader = DataLoader(
      valid_dataset,
      batch_size=batch_size_val,
      shuffle=False,
      num_workers=num_workers,
      drop_last=True,
    )

    # And excpect to get an OrderedDict of loaders
    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders


def select_model(model_name):
    
    # FPN_mit  FPN_res34   FPN_mob Unet_res34  Unet_mob  Unet_res101   MAnet_res34   MAnet_mob  MAnet_res101
    # add more models if necessary from https://github.com/qubvel/segmentation_models.pytorch (0.3.3)
    if model_name == "FPN_mit":
        model = smp.FPN(encoder_name="mit_b0", encoder_weights="imagenet", classes=1)

    elif model_name == "FPN_res34":
        model = smp.FPN(encoder_name="resnet34", encoder_weights="imagenet", classes=1)

    elif model_name == "FPN_mob":
        model = smp.FPN(encoder_name="timm-mobilenetv3_large_100", encoder_weights="imagenet", classes=1)

    elif model_name == "Unet_res34":
        model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=1)

    elif model_name == "Unet_mob":
        model = smp.Unet(encoder_name="timm-mobilenetv3_large_100", encoder_weights="imagenet", classes=1)

    elif model_name == "Unet_res101":
        model = smp.Unet(encoder_name="resnet101", encoder_weights="imagenet", classes=1)

    elif model_name == "MAnet_res34":
        model = smp.MAnet(encoder_name="resnet34", encoder_weights="imagenet", classes=1) 

    elif model_name == "MAnet_mob":
        model = smp.MAnet(encoder_name="timm-mobilenetv3_large_100", encoder_weights="imagenet", classes=1) 

    elif model_name == "MAnet_res101": 
        model = smp.MAnet(encoder_name="resnet101", encoder_weights="imagenet", classes=1) 
        
    return model


def main(num_epochs, ROOT, model_name, size, uptype, path_model): 
    
    # hyperparameters
    learning_rate = 0.001
    encoder_learning_rate = 0.0005
    batch_size = 8
    
    # read data
    if uptype == "":
        train_img = "train_"+size+"/images"
        
    else:
        train_img = "train_"+size+"/"+uptype+"/images"
        
    train_gt = "train_"+size+"/gt"
    
    train_image_path = ROOT / train_img
    train_mask_path = ROOT / train_gt
    test_image_path = ROOT / "val/images"
    test_mask_path = ROOT / "val/gt"

    ALL_IMAGES_tr = sorted(train_image_path.glob("*.png"))
    ALL_MASKS_tr = sorted(train_mask_path.glob("*.png"))
    ALL_IMAGES_val = sorted(test_image_path.glob("*.png")) 
    ALL_MASKS_val = sorted(test_mask_path.glob("*.png"))
    
    if uptype == "":
        train_transforms = compose([
                                    resize_transforms(), 
                                    hard_transforms(), 
                                    post_transforms()])
    else:
        train_transforms = compose([
                                    resize_transforms(), 
                                    hard_transforms_SR(), 
                                    post_transforms()])

    # set transforms
    valid_transforms = compose([resize_transforms(), post_transforms()])
    
    # load data
    loaders = get_loaders(
                        images_tr=ALL_IMAGES_tr,
                        masks_tr=ALL_MASKS_tr,
                        images_val=ALL_IMAGES_val,
                        masks_val=ALL_MASKS_val,
                        random_state=SEED,
                        train_transforms_fn=train_transforms,
                        valid_transforms_fn=valid_transforms,
                        batch_size=batch_size)
    
    # choose model
    model = select_model(model_name)
    
    # multiple criterions
    criterion = {
        "dice": DiceLoss(),
        "iou": IoULoss(),
        "bce": nn.BCEWithLogitsLoss()
    }
    
    
    # Since we use a pre-trained encoder, we will reduce the learning rate on it.
    layerwise_params = {"encoder*": dict(lr=encoder_learning_rate, weight_decay=0.00003)}

    # This function removes weight_decay for biases and applies our layerwise_params
    model_params = utils.process_model_params(model, layerwise_params=layerwise_params)

    # Catalyst has new SOTA optimizers out of box
    base_optimizer = RAdam(model_params, lr=learning_rate, weight_decay=0.0003)
    optimizer = Lookahead(base_optimizer)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)
    
    # this variable will be used in `runner.train` and by default we disable FP16 mode
    is_fp16_used = False

    device = utils.get_device()
    print(f"device: {device}")

    if is_fp16_used:
        fp16_params = dict(opt_level="O1") # params for FP16
    else:
        fp16_params = None

    # by default SupervisedRunner uses "features" and "targets",
    # in our case we get "image" and "mask" keys in dataset __getitem__
    runner = SupervisedRunner(device=device, input_key="image", input_target_key="mask")
    
    callbacks = [
    # Each criterion is calculated separately.
    CriterionCallback(
        input_key="mask",
        prefix="loss_dice",
        criterion_key="dice"
    ),
    CriterionCallback(
        input_key="mask",
        prefix="loss_iou",
        criterion_key="iou"
    ),
    CriterionCallback(
        input_key="mask",
        prefix="loss_bce",
        criterion_key="bce"
    ),

    # And only then we aggregate everything into one loss.
    MetricAggregationCallback(
        prefix="loss",
        mode="weighted_sum", # can be "sum", "weighted_sum" or "mean"
        # because we want weighted sum, we need to add scale for each loss
        metrics={"loss_dice": 1.0, "loss_iou": 1.0, "loss_bce": 0.8},
    ),

    # metrics
    DiceCallback(input_key="mask"),
    IouCallback(input_key="mask"),
    # visualization
    DrawMasksCallback(output_key='logits',
                      input_image_key='image',
                      input_mask_key='mask',
                      summary_step=50
        )
    ]

    # if runner has an error that says "lookahead.py doesnt have a _optimizer_step_pre_hooks, use the lookahead.py to replace the one in the lib (the path should be mentioned in bug report)"
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/lookahead.py
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        # our dataloaders
        loaders=loaders,
        # We can specify the callbacks list for the experiment;
        callbacks=callbacks,
        # path to save logs in the same folder of output models
        logdir=path_model,
        num_epochs=num_epochs,
        # save our best checkpoint by IoU metric
        main_metric="iou",
        # IoU needs to be maximized.
        minimize_metric=False,
        # for FP16. It uses the variable from the very first cell
        fp16=fp16_params,
        # prints train logs
        verbose=True,
    )
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--data', default=None)
    parser.add_argument('--model', default=None, help="FPN_mit  FPN_res34   FPN_mob Unet_res34  Unet_mob  Unet_res101   MAnet_res34   MAnet_mob  MAnet_res101")
    parser.add_argument('--upsample', default="1024", help="1024 or SR") 
    parser.add_argument('--size', default="small", help="small or large, if there is only one training dataset, you can simply add a character or number to tag the train folder, e.g., train1") 
    parser.add_argument('--uptype', default="", help="nearest bilinear EDSR") 
    parser.add_argument('--model_type', default="SAM", help="SAM, SS") 
    
    # path_data = "path of your data folder" # e.g. "/home/usr/Data"
    path_data = "/home/yunya/anaconda3/envs/Data"
    
    args = parser.parse_args()
    
    data_name = args.data
    model_name = args.model
    num_epoch = args.epoch
    upsample = args.upsample
    uptype = args.uptype
    model_type = args.model_type
    size = args.size
    
    path_model = os.path.join("save_model", data_name, size, upsample, uptype, model_name)
    pathlib.Path(path_model).mkdir(parents=True, exist_ok=True)
        
    # ROOT: the root of image and label data
    ROOT = Path(path_data+"/"+data_name+"/"+model_type+"/"+upsample)

    main(num_epoch, ROOT, model_name, size, uptype, path_model)
    
    
