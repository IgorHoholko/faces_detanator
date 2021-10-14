"""
 File name   : io.py
 Description : description

 Date created : 24.09.2021
 Author:  Ihar Khakholka
"""

import yaml
import json
from typing import Tuple, List


def read_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        f_str = f.read()
        file = yaml.load(f_str, Loader=yaml.FullLoader)
    return file


def read_file(path: str) -> List[str]:
    with open(path) as f:
        data = f.readlines()
    return data

def dump_json(fp: str, data: dict):
    with open(fp, 'w') as f:
        json.dump(data, f)

def load_json(fp: str) -> dict:
    with open(fp, 'r') as f:
        data = json.load(f)
    return data

def parse_line(line: str) -> Tuple[str, List[list], List[list]]:
    detections = []
    landmarks = []

    det_size = 1 + 4 + 10

    img = line.split(' ')[0]
    count = int(line.split(' ')[1])
    line = line.split("$d")[1].strip().split(' ')

    for i in range(count):
        det_raw = line[i * det_size: (i + 1) * det_size]
        points = [float(p) for p in det_raw[5:]]
        x1 = min(int(det_raw[1]), int(det_raw[3]))
        x2 = max(int(det_raw[1]), int(det_raw[3]))
        y1 = min(int(det_raw[2]), int(det_raw[4]))
        y2 = max(int(det_raw[2]), int(det_raw[4]))
        score = float(det_raw[0])

        detections.append([x1, y1, x2, y2, score])
        landmarks.append(points)

    return img, detections, landmarks


def collect_detections(detections_log: List[str], prefix_path: str = None) -> dict:
    output = dict()

    for line in detections_log:
        img, detections, landmarks = parse_line(line)

        if prefix_path:
            img = img[img.find(prefix_path):]

        output[img] = [detections, landmarks]

    return output


