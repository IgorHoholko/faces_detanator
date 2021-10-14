"""
 File name   : draw_output.py
 Description : description

 Date created : 25.09.2021
 Author:  Ihar Khakholka
"""


import argparse
import os
import shutil

from tqdm import tqdm
import cv2

from utils.draw_box import BoxesDrawer
from utils.io import load_json, dump_json
from utils.transforms import log_file_to_internal


def _parse_args():
    parser = argparse.ArgumentParser(); add = parser.add_argument
    add('-i', "--input", type=str, help="Path to output")
    add('-o', "--output", type=str, default='./output/imgs', help="Path to save results")
    add('-l', "--limit", type=str, default=None, help='Limit number of images to process')
    return parser.parse_args()


if __name__ == '__main__':

    args = _parse_args()

    if '.json' in args.input:
        data = load_json(args.input)
    elif '.txt' in args.input:
        data = log_file_to_internal(args.input)
    else:
        raise NotImplementedError(f"Not known format: {args.input}!!!")

    drawer = BoxesDrawer()
    if os.path.exists(args.output):
        if input(f"{args.output} folder already exists. Remove and recreate? (yes/no)") == 'yes':
            shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)

    for ipath, detections in tqdm(data.items(), desc="Images annotating.."):
        img = cv2.imread(ipath)
        iname = ipath.split('/')[-1]

        boxes = [det['box'] for det in detections]
        scores = [det['score'] for det in detections]
        landmarks = [det['landmarks'] for det in detections]
        labels = ['Face']*len(scores)
        img = drawer.draw(img, boxes, scores, labels, landmarks)

        cv2.imwrite(os.path.join(args.output, iname), img)
