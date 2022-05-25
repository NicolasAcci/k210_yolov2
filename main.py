import time

import numpy as np
np.random.seed(111)
import os
import json
from yolo.frontend import create_yolo, get_object_labels
import warnings
import shutil
import argparse




def setup_training(config_file):
    print(config_file)
    
    with open(config_file,encoding='utf-8') as config_buffer:
        config = json.loads(config_buffer.read())
    dirname = config['train']['saved_folder']
    if os.path.isdir(dirname):
        print("{} is already exists. Weight file in directory will be overwritten".format(dirname))
    else:
        print("{} is created.".format(dirname, dirname))
        os.makedirs(dirname)
    print("Weight file and Config file will be saved in \"{}\"".format(dirname))
    shutil.copyfile(config_file, os.path.join(dirname, "config.json"))
    return config, os.path.join(dirname, "weights.h5")


def run(configs_url,alphaa):
    config_file = configs_url
    config, weight_file = setup_training(config_file)
    if config['train']['is_only_detect']:
        labels = ["object"]
    else:
        if config['model']['labels']:
            labels = config['model']['labels']
        else:
            labels = get_object_labels(config['train']['train_annot_folder'])

    print(labels)

    # 1. Construct the model 
    yolo = create_yolo(config['model']['architecture'],
                    labels,
                    config['model']['input_size'],
                    config['model']['anchors'],
                    config['model']['coord_scale'],
                    config['model']['class_scale'],
                    config['model']['object_scale'],
                    config['model']['no_object_scale'],
                    alpha=alphaa)

    # 2. Load the pretrained weights (if any) 
    if alphaa=="1.0":
        weights='models/mobilenet_1_0_224_tf_no_top.h5'
    elif alphaa=="0.75":
        weights='models/mobilenet_7_5_224_tf_no_top.h5'
    elif alphaa=="0.5":
        weights='models/mobilenet_5_0_224_tf_no_top.h5'
    elif alphaa=="0.25":
        weights='models/mobilenet_2_5_224_tf_no_top.h5'
    yolo.load_weights(weights, by_name=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 3. actual training 
        yolo.train(config['train']['train_image_folder'],
                config['train']['train_annot_folder'],
                config['train']['actual_epoch'],
                weight_file,
                config["train"]["batch_size"],
                config["train"]["jitter"],
                config['train']['learning_rate'], 
                config['train']['train_times'],
                config['train']['valid_times'],
                config['train']['train_image_folder'],
                config['train']['train_annot_folder'],
                config['train']['first_trainable_layer'],
                config['train']['is_only_detect'],
                (len(config['model']["labels"])+5)*5,
                config['model']['anchors'],
                w_name=config['model']['architecture'],
                alpha=alphaa,
                lable=config['model']['labels']
                )

if __name__=="__main__":
    t1= time.time()
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument("-f","--file",help="configs.json file url",default="configs\configs.json")
    parser.add_argument("-a","--alpha",help="alpha 1.0,0.75,0.5,0.25",default="0.75")
    args = parser.parse_args()
    if os.path.exists(args.file)==False:
        print("没有找到配置文件,请重新选择.")
    else:
        f = open(args.file, "r",encoding='utf-8')
        setting = json.load(f)
        run(args.file,args.alpha)
        print("训练完成时间：",time.time()-t1)
        print("End of training!")


