import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import glob
from skimage import io
import pathlib
from pprint import pprint

import matplotlib.pyplot as plt
import matplotlib.colors as colors


def read_pred_gt_data(path_pred, path_gt): 
    
    # read data from path
    pred = io.imread(path_pred)
    gt = io.imread(path_gt)
    
    # process pred and gt
    new_width = min(pred.shape[0], gt.shape[0]) # perhaps one pixel shift, so to make their shape the same
    new_height = min(pred.shape[1], gt.shape[1])

    pred = pred[:new_width, :new_height]
    gt = gt[:new_width, :new_height]

    # turn 255 in gt or pred to 1
    pred[pred==255] = 1
    gt[gt==255] = 1
    
    return pred, gt


def visualize_diff(pred, gt, thres, path_out):
    
    # diff = gt - pred
    colour_dict = {-1: colors.to_rgb('crimson'), # false positive
                    0: colors.to_rgb('gainsboro'),
                    1: colors.to_rgb('blue')} # false negative
    
    colours_rgb = [colour_dict[i] for i in [-1, 0, 1]]
    colours_rgb = colors.ListedColormap(colours_rgb)

    # here, the values of gt and pred should be 0 and 1.
    # the values of diff: -1 (false positive), 0, 1 (false negative)
    diff = gt - pred 
    
    plt.figure(figsize=(50, 50))
    plt.axis('off')
    
    path_out_diff = os.path.join(path_out, "diff.png")
    plt.imshow(diff, cmap=colours_rgb, vmin=-1, vmax=1)
    plt.savefig(path_out_diff, bbox_inches='tight')
    
    path_out_gt = os.path.join(path_out, "gt.png")
    plt.imshow(gt)
    plt.savefig(path_out_gt, bbox_inches='tight')
    
    path_out_pred = os.path.join(path_out, "mask_pred_binary"+str(thres)+".png")
    plt.imshow(pred)
    plt.savefig(path_out_pred, bbox_inches='tight')


def calculate_iou(pred, gt):

    # IoU calculation = intersection / uniion
    intersection = np.logical_and(pred, gt)
    union = np.logical_or(pred, gt)
    iou_score = np.sum(intersection) / np.sum(union)
    
    return iou_score


def calculate_precision(pred, gt):
    
    # precision = TP (true predicted postive) / AP (all positive in gt)
    TP = np.sum(np.logical_and(pred, gt))
    AP = np.sum(pred)
    precision_score = TP / AP
    
    return precision_score


def calculate_recall(pred, gt):
    
    # recall = TP / PP (predicted postive)
    TP = np.sum(np.logical_and(pred, gt))
    PP = np.sum(gt) 
    recall_score = TP / PP
    
    return recall_score


def calculate_f1(pred, gt):
    
    # f1 = 2 * precision * recall / (precision + recall)
    precision = calculate_precision(pred, gt)
    recall = calculate_recall(pred, gt)
    f1_score = 2*precision*recall / (precision+recall)
    
    return f1_score


def evaluate_metrics(pred, gt):
    
    eval_dict = {}
    
    recall = calculate_recall(pred, gt)
    precision = calculate_precision(pred, gt)
    f1= calculate_f1(pred, gt)
    iou = calculate_iou(pred, gt)
    
    eval_dict["recall"] = recall
    eval_dict["precision"] = precision
    eval_dict["f1"] = f1
    eval_dict["iou"] = iou
    
    return eval_dict

def save_results_as_excel(results, path_out):
    # results: a dictionary with accuracy metrics values
    
        # convert into dataframe
    df = pd.DataFrame(data=results, index=[1])

    #convert into excel
    df.to_csv(path_out+"/results.cvs", index=False)
    
    
# main function
def main(path_pred, path_gt, path_out, thres):
    
    # get predicted binary mask and gt mask
    pred, gt = read_pred_gt_data(path_pred, path_gt)
    
    # evaluate 
    eval_result = evaluate_metrics(pred, gt)
    
    # save evaluation results as excel
    save_results_as_excel(eval_result, path_out)
    
    # save gt and predicted binary mask
    visualize_diff(pred, gt, thres, path_out)
    
    pprint(eval_result)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=None)
    parser.add_argument('--model', default=None, help="FPN_mit  FPN_res34   FPN_mob Unet_res34  Unet_mob  Unet_res101   MAnet_res34   MAnet_mob  MAnet_res101")
    parser.add_argument('--upsample', default="1024", help="1024 or SR") 
    parser.add_argument('--size', default="small", help="small or large") 
    parser.add_argument('--uptype', default="", help="nearest bilinear EDSR") 
    parser.add_argument('--thres', default=0.5, help="threshold to determine the binary map") 
    
    # path_data = "path of your data folder" # e.g. "/home/usr/Data"
    path_data = "/home/yunya/anaconda3/envs/Data"
    
    args = parser.parse_args()
    
    data_name = args.data
    model_name = args.model
    upsample = args.upsample
    size = args.size
    uptype = args.uptype
    thres = args.thres
    
    # get path of gt
    gt_path_ = os.path.join(path_data, data_name, "raw", "test", "gt")
    gt_path = glob.glob(gt_path_ + "/*.tif")[0]
    
    # get path of prediction
    pred_path = os.path.join("outputs", data_name, size, upsample, uptype, model_name, "pred_mask_bin"+str(thres)+".tif")
    path_out_diff = os.path.join("outputs", data_name, size, upsample, uptype, model_name)
    
    # define path of output difference map
    pathlib.Path(path_out_diff).mkdir(parents=True, exist_ok=True)
    print(path_out_diff)

    # run the main function
    main(pred_path, gt_path, path_out_diff, thres)