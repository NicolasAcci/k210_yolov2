import argparse
import json
from cv2 import cv2
import numpy as np
from yolo.frontend import create_yolo
from yolo.backend.utils.box import draw_scaled_boxes
from yolo.backend.utils.annotation import parse_annotation
from yolo.backend.utils.eval.fscore import count_true_positives, calc_score

from pascal_voc_writer import Writer
from shutil import copyfile
import os
import yolo

parser = argparse.ArgumentParser(description="Demo of argparse")
parser.add_argument("-f","--file",help="model file url",default="models/yolov2_mobile_for_k210.tflite")
parser.add_argument("-c","--conf",help="config file url",default="configs/configs.json")
args = parser.parse_args()

evaluation_object = "test"
DEFAULT_CONFIG_FILE = args.conf
DEFAULT_WEIGHT_FILE = args.file
DEFAULT_THRESHOLD = 0.4

with open(DEFAULT_CONFIG_FILE, encoding='UTF-8') as config_buffer:
    config = json.loads(config_buffer.read())

def create_ann(filename, image, boxes, labels,label_list):
    if not os.path.exists('test/imgs/'):
        os.makedirs('test/imgs/')
    if not os.path.exists('test/ann/'):
        os.makedirs('test/ann/')
    copyfile(os.path.join(config['train']['test_folder'], filename), 'test/imgs/' + filename)
    
    writer = Writer(os.path.join(config['train']['test_folder'], filename), 224, 224)
    writer.addObject(label_list[labels[0]], boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3])
    name = filename.split('.')
    writer.save('test/ann/' + name[0] + '.xml')

# 2. create yolo instance & predict
def file_name(file_dir):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.jpg':  
                L.append(os.path.join(root, file))  
    return L 
f=file_name("test/Yolo")
for i in f:
    os.remove(i)
yolo = create_yolo(config['model']['architecture'],
                   config['model']['labels'],
                   config['model']['input_size'],
                   config['model']['anchors'])
yolo.load_weights(DEFAULT_WEIGHT_FILE,by_name=True)

# 3. read image
write_dname = "test/Yolo"
if not os.path.exists(write_dname): os.makedirs(write_dname)
annotations = parse_annotation(config['train']['valid_annot_folder'],
                               config['train']['valid_image_folder'],
                               config['model']['labels'],
                               is_only_detect=config['train']['is_only_detect'])

#n_true_positives = 0
#n_truth = 0
#n_pred = 0
#for i in range(len(annotations)):
for filename in os.listdir(config['train']['test_folder']):
    img_path = os.path.join(config['train']['test_folder'], filename)
    img_fname = filename
    image = cv2.imread(img_path)

    boxes, probs = yolo.predict(image, float(DEFAULT_THRESHOLD))
    labels = np.argmax(probs, axis=1) if len(probs) > 0 else [] 

    # 4. save detection result
    image = draw_scaled_boxes(image, boxes, probs, config['model']['labels'])
    output_path = os.path.join(write_dname,os.path.split(img_fname)[-1])
    label_list = config['model']['labels']
    cv2.imwrite(output_path, image)
    print("{}-boxes are detected. {} saved.".format(len(boxes), output_path))
    if len(probs) > 0:
        cv2.imwrite(output_path, image)

