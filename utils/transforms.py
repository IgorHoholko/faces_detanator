"""
 File name   : transforms.py
 Description : description

 Date created : 25.09.2021
 Author:  Ihar Khakholka
"""

from typing import List
from collections import defaultdict


def log_to_json(data: List[str]) -> dict:
    output = defaultdict(list)

    det_size = 1 + 4 + 10

    for line in data:
        img = line.split(' ')[0]
        count = int(line.split(' ')[1])
        line = line.split("$d")[1].strip().split(' ')

        if not count:
            continue

        for i in range(count):
            det_raw = line[i * det_size: (i + 1) * det_size]
            landmarks = [float(p) for p in det_raw[5:]]
            x1 = min(int(det_raw[1]), int(det_raw[3]))
            x2 = max(int(det_raw[1]), int(det_raw[3]))
            y1 = min(int(det_raw[2]), int(det_raw[4]))
            y2 = max(int(det_raw[2]), int(det_raw[4]))
            score = float(det_raw[0])

            detection = {"box": [x1,y1,x2,y2], "score": score, "landmarks": landmarks}

            output[img].append(detection)
    return output

def log_file_to_internal(log_file: str) -> dict:
    with open(log_file) as f:
        data = f.readlines()
    return log_to_json(data)

