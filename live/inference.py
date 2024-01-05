from pycocotools.coco import COCO
import torch
# faster rcnn model이 포함된 library
import model as module_arch
import collections
import argparse

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from parse_config import ConfigParser

import data_loader as module_data
import pandas as pd
from tqdm import tqdm

def inference_fn(test_data_loader, model, device):
    outputs = []
    for images in tqdm(test_data_loader):
        # gpu 계산을 위해 image.to(device)
        images = list(image.to(device) for image in images)
        output = model(images)
        for out in output:
            outputs.append({'boxes': out['boxes'].tolist(), 'scores': out['scores'].tolist(), 'labels': out['labels'].tolist()})
    return outputs

def main(config):
    annotation = config["tester"]["annotation"] # annotation 경로
    check_point = config["tester"]["checkpoint_path"]
    score_threshold = 0.05
    
    test_data_loader = config.init_data_loader("data_loader", module_data)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    
    # torchvision model 불러오기
    model = config.init_obj("arch", module_arch)
    model.to(device)
    model.load_state_dict(torch.load(check_point, map_location=device))
    model.eval()
    
    outputs = inference_fn(test_data_loader, model, device)
    prediction_strings = []
    file_names = []
    coco = COCO(annotation)

    # submission 파일 생성
    for i, output in enumerate(outputs):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > score_threshold: 
                # label[1~10] -> label[0~9]
                prediction_string += str(label-1) + ' ' + str(score) + ' ' + str(box[0]) + ' ' + str(
                    box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv('./submit/faster_rcnn_torchvision_submission.csv', index=None)
    print(submission.head())
    
if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Practical Pytorch")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: None)",
    )

    CustomArgs = collections.namedtuple("CustomArgs", ["flags", "type", "target"])
    options = [
        CustomArgs(
            flags=["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"
        ),
        CustomArgs(
            flags=["--bs", "--batch_size"],
            type=int,
            target="data_loader;args;batch_size",
        ),
    ]

    config = ConfigParser.from_args(args, options)
    main(config)    