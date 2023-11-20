# the main change is to reduce the size from 1024 to 224 which is the size during model training

import sys
import os

# Get the parent directory of the current script
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the Python path
sys.path.append(parent_directory)

import argparse
import numpy as np
import yaml
from tqdm import tqdm
import cv2
import scipy
import imagecodecs
from skimage.transform import resize
from skimage import io
import glob
import pathlib
import geopandas as gpd
import rasterio

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.distributed as dist
from torchvision.transforms import functional as F
import segmentation_models_pytorch as smp

import albumentations as albu

# import datasets
# import models
import utils
from common_ft import *


torch.distributed.init_process_group(backend='nccl')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class Compose(object):
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, image, ann=None):
#         if ann is None:
#             for t in self.transforms:
#                 image = t(image)
#             return image
#         for t in self.transforms:
#             image, ann = t(image, ann)
#         return image, ann


# class Resize(object):
#     def __init__(self, image_height, image_width, ann_height, ann_width):
#         self.image_height = image_height
#         self.image_width = image_width
#         self.ann_height = ann_height
#         self.ann_width = ann_width

#     def __call__(self, image, ann):
#         image = resize(image, (self.image_height, self.image_width))
#         image = np.array(image, dtype=np.float32) / 255.0

#         sx = self.ann_width / ann['width']
#         sy = self.ann_height / ann['height']
#         ann['junc_ori'] = ann['junctions'].copy()
#         ann['junctions'][:, 0] = np.clip(ann['junctions'][:, 0] * sx, 0, self.ann_width - 1e-4)
#         ann['junctions'][:, 1] = np.clip(ann['junctions'][:, 1] * sy, 0, self.ann_height - 1e-4)
#         ann['width'] = self.ann_width
#         ann['height'] = self.ann_height
#         ann['mask_ori'] = ann['mask'].copy()
#         ann['mask'] = cv2.resize(ann['mask'].astype(np.uint8), (int(self.ann_width), int(self.ann_height)))

#         return image, ann


# class ResizeImage(object):
#     def __init__(self, image_height, image_width):
#         self.image_height = image_height
#         self.image_width = image_width

#     def __call__(self, image, ann=None):
#         image = resize(image, (self.image_height, self.image_width))
#         image = np.array(image, dtype=np.float32) / 255.0
#         if ann is None:
#             return image
#         return image, ann


# class ToTensor(object):
#     def __call__(self, image, anns=None):
#         if anns is None:
#             return F.to_tensor(image)

#         for key, val in anns.items():
#             if isinstance(val, np.ndarray):
#                 anns[key] = torch.from_numpy(val)
#         return F.to_tensor(image), anns
    
# def inference_transforms():
#     # we use ImageNet image normalization
#     # and convert it to torch.Tensor
#     return [albu.Normalize(), ToTensor()]
  
def compose(transforms_to_compose):
    # combine all augmentations into single pipeline
    result = albu.Compose([
      item for sublist in transforms_to_compose for item in sublist
    ])
    return result
    

def inference_image(image, model):
    
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
                    # albu.Normalize()
                ])
    # transform = compose([inference_transforms()])

    h_stride, w_stride = 160, 160
    h_crop, w_crop = 224, 224
    h_img, w_img, _ = image.shape
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    pred_whole_img = np.zeros([h_img, w_img], dtype=np.float32)
    count_mat = np.zeros([h_img, w_img])
    juncs_whole_img = []

    patch_weight = np.ones((h_crop + 2, w_crop + 2))
    patch_weight[0,:] = 0
    patch_weight[-1,:] = 0
    patch_weight[:,0] = 0
    patch_weight[:,-1] = 0

    patch_weight = scipy.ndimage.distance_transform_edt(patch_weight)
    patch_weight = patch_weight[1:-1,1:-1]

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)

            crop_img = image[y1:y2, x1:x2, :]
            crop_img = crop_img.astype(np.float32)
            crop_img_tensor = transform(crop_img)[None].to(device)

            meta = {
                'height': crop_img.shape[0],
                'width': crop_img.shape[1],
                'pos': [x1, y1, x2, y2]
            }

            with torch.no_grad():
                mask_pred = model.predict(crop_img_tensor)
                mask_pred = torch.sigmoid(mask_pred)
                mask_pred = mask_pred.cpu().numpy().copy()[0,0]

            mask_pred *= patch_weight
            pred_whole_img += np.pad(mask_pred,
                                ((int(y1), int(pred_whole_img.shape[0] - y2)),
                                (int(x1), int(pred_whole_img.shape[1] - x2))))
            count_mat[y1:y2, x1:x2] += patch_weight

    pred_whole_img = pred_whole_img / count_mat

    return pred_whole_img


def get_binary_mask(pred_mask, thres): 
    
    binar_mask = pred_mask > thres
    return binar_mask


def save_fig(data, data_name, path_out):

    # create figure
    plt.figure(figsize=(50, 50))
    plt.axis('off')
    
    # save data
    path_out_ = os.path.join(path_out, data_name+".png")
    plt.imshow(data)
    plt.savefig(path_out_, bbox_inches='tight')
    

def tiff_to_shp(tiff_path, output, simplify_tolerance=0.001, **kwargs):
    """Convert a tiff file to a shapefile.
    Args:
        tiff_path (str): The path to the tiff file.
        output (str): The path to the shapefile.
        simplify_tolerance (float, optional): The maximum allowed geometry displacement.
            The higher this value, the smaller the number of vertices in the resulting geometry.
    """
    raster_to_shp(tiff_path, output, simplify_tolerance=simplify_tolerance, **kwargs)


def save_predicted_probability_mask_shapefile(img_path, path_out, pred_whole_img):

    thres = 0.5
    
    # read spatial information from test image
    with rasterio.open(img_path) as src:
        ras_meta = src.profile
        crs = src.crs # for vector
        ras_meta["count"] = 1 # for raster
        ras_meta["dtype"] = "float32"

    pred_whole_img_prob = np.expand_dims(pred_whole_img[...], axis=0)
    pred_whole_img_bin = pred_whole_img_prob > thres
    
    # save probability as png
    save_fig(pred_whole_img, "mask_prob", path_out)

    # save probability map
    path_mask_out = path_out+"/pred_mask_prob.tif"
    with rasterio.open(path_mask_out, 'w', **ras_meta) as dst:
        dst.write(pred_whole_img_prob)

    # save predicted binary mask
    path_mask_out_bin = path_out+"/pred_mask_bin"+str(thres)+".tif"
    with rasterio.open(path_mask_out_bin, 'w', **ras_meta) as dst:
        dst.write(pred_whole_img_bin)

    # save polygons as shapefile
    path_shp_out = path_out+"/pred_mask_bin"+str(thres)+".shp"
    tiff_to_shp(path_mask_out_bin, path_shp_out)
    
    
def select_model(model_name):
    
    # FPN_mit  FPN_res34   FPN_mob Unet_res34  Unet_mob  Unet_res101   MAnet_res34   MAnet_mob  MAnet_res101
    # selected models from https://github.com/qubvel/segmentation_models.pytorch (0.3.3)
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

    
# main function
def main(data_name, model_name, size, upsample, path_data, path_model, path_out):
    
    # read image for testing (a large geotiff data)
    img_path_ = path_data + "/" + data_name + "/raw/test/images"
    img_path = glob.glob(img_path_ + "/*.tif")[0]
    image = io.imread(img_path)
    image = (image - image.min()) / (image.max() - image.min())
    save_fig(image, "image", path_out)
    
    # read ground truth (gt) data 
    gt_path_ = path_data + "/" + data_name + "/raw/test/gt"
    gt_path = glob.glob(gt_path_ + "/*.tif")[0]
    gt = io.imread(gt_path)
    
    # get model
    model = select_model(model_name)
    checkpoint = torch.load(path_model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.cuda()
    model.eval()
    
    # probability map
    prob_mask = inference_image(image, model)
    print("Predicted probability map.")
    
    # binary mask based on probability map and input threshold
    thres = 0.5
    binar_mask = get_binary_mask(prob_mask, thres)
    print("Get predicted binary mask with a threshold of {}".format(thres))
    
    # save probability map, binary mask, shapefile
    save_predicted_probability_mask_shapefile(img_path, path_out, prob_mask)
    print("Save predicted probability map, binary map and shapefile.")
    

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=None)
    parser.add_argument('--model', default=None, help="FPN_mit  FPN_res34   FPN_mob Unet_res34  Unet_mob  Unet_res101   MAnet_res34   MAnet_mob  MAnet_res101")
    parser.add_argument('--upsample', default="1024", help="1024 or SR") 
    parser.add_argument('--size', default="small", help="small or large") 
    parser.add_argument('--uptype', default="", help="nearest bilinear EDSR") 
    
    # path_data = "path of your data folder" # e.g. "/home/usr/Data"
    path_data = "/home/yunya/anaconda3/envs/Data"
    
    args = parser.parse_args()
    
    data_name = args.data
    model_name = args.model
    upsample = args.upsample
    size = args.size
    uptype = args.uptype
    
    # if uptype == "":
    #     path_model_base = os.path.join("save_model", data_name, size, upsample, model_name)
    #     path_out = os.path.join("outputs", data_name, size, upsample, model_name)
    # else:
    #     path_model_base = os.path.join("save_model", data_name, size, upsample, uptype, model_name)
    #     path_out = os.path.join("outputs", data_name, size, upsample, uptype, model_name)
        
    path_model_base = os.path.join("save_model", data_name, size, upsample, uptype, model_name)
    path_out = os.path.join("outputs", data_name, size, upsample, uptype, model_name)
        
    # there are multiple outputs in checkpoints, e.g., last_full.pth, best.pth, last.pth...
    path_model = os.path.join(path_model_base, "checkpoints", "best_full.pth")
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)
    
    main(data_name, model_name, size, upsample, path_data, path_model, path_out)
    
   