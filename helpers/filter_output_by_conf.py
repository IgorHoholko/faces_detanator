"""
 File name   : filter_output_by_conf.py
 Description : description

 Date created : 25.09.2021
 Author:  Ihar Khakholka
"""

import argparse
import os

from utils.io import load_json, dump_json


def _parse_args():
    parser = argparse.ArgumentParser(); add = parser.add_argument
    add('-i', "--input", type=str, help="Path to folders you want to process")
    add('-t', '--thresh_conf', type=float, help="Detections with confidence lower than <thresh_conf> will be removed")

    add('-o', "--output_path", type=str, default='./output', help="Path to save results")
    add('-f', "--filename", type=str, default=None, help="Name of output file. By default name will be taken from <input>")
    return parser.parse_args()


if __name__ == '__main__':

    args = _parse_args()

    if '.json' in args.input:
        data = load_json(args.input)
        for ipath, detections in data.items():
            data[ipath] = list(filter(lambda d: d['score'] >= args.thresh_conf, detections))

        file_name = args.filename or args.input.replace('.json', f'_filtered_{args.thresh_conf}.json')
        save_path = os.path.join(args.output, file_name)
        dump_json(save_path, data)

    else:
        raise NotImplementedError("Not implemented for 'log' format")