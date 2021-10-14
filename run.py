import argparse
import os
import glob
import addict
from typing import List
from collections import defaultdict
import logging
import shutil

from tqdm import tqdm
import numpy as np

from utils.io import read_yaml, read_file, dump_json, collect_detections
from utils.helpers import run_command, filter_prefix
from utils.voting import aggregate_detections
from utils.logger import init_logger


SPLIT_CHAR = "P1K2-RSK12aDf215Zzz"
SETTINGS_PATH = "settings.yaml"


def _parse_args():
    parser = argparse.ArgumentParser(); add = parser.add_argument
    add('-i', "--input", type=str, help="Path to folders you want to process")
    add('-p', "--prefix", type=str, default=None, help="Prefix path to remove from images when save. "
                                                       "Example: <ipath>=a/b/c/d.jpg, <prefix>=a/b/c, <resut_ipath>=d.jpg")
    add('-o', "--output_path", type=str, default='./output', help="Path to save results")
    add('-f', "--filename", type=str, default=None, help="Name of output file. By default name will be taken from <input>")
    add('--keep_temp', action="store_true", default=False, help="If specified - temp not aggregated detectors metadata will be saved")
    add('--save_format', type=str, default="dataset", help="Output format: <dataset> or <log>")
    return parser.parse_args()

def _sanity_check(args, settings: addict.Dict) -> None:
    if len(settings.thresh_iou) != len(settings.min_votes):
        raise ValueError("<settings.thresh_iou> and <settings.min_votes> must have same size!!!")

    for detector, params in settings.detectors.items():
        if 'dir' not in params:
            raise ValueError(f"Specify <dir> for ({detector}) detector in settings.yaml!!!")
        if "Dockerfile" not in os.listdir(os.path.join("./detectors", params.dir)):
            raise ValueError(f"Create Dockerfile for ({detector}) detector!!!")
        if "entrypoint.sh" not in os.listdir(os.path.join("./detectors", params.dir)):
            raise ValueError(f"Create entrypoint.sh for ({detector}) detector!!!")
    if args.save_format not in ('dataset', 'log'):
        raise ValueError(f"<save_format> can be eather 'dataset' or 'log', not ({args.save_format})!!!")
    if args.prefix and args.prefix not in args.input:
        raise ValueError(f"Prefix ({args.prefix}) doesn't exist in input path ({args.input})!!!")

if __name__ == '__main__':

    args = _parse_args()
    logger = init_logger('info')
    settings = addict.Dict(read_yaml(SETTINGS_PATH))

    try:
        _sanity_check(args, settings)
    except Exception as e:
        raise RuntimeError(e)

    detectors = list(settings.detectors.keys())

    # BUILD
    logger.log(logging.INFO, "Start building ...")
    for detector in tqdm(detectors, desc="Building images"):
        dir_ = settings.detectors[detector].dir
        tag = dir_.lower()
        cmd = f"docker build --rm --tag {tag} ./detectors/{dir_}"
        logger.log(logging.INFO, cmd)
        run_command(cmd)


    # INFER
    logger.log(logging.INFO, "Start infering ...")
    cwd = os.getcwd()

    dataset = SPLIT_CHAR.join( args.input.split('/') )

    for detector in tqdm(detectors):
        save_folder = f"{cwd}/temp/{dataset}/{detector}"
        os.makedirs(save_folder, exist_ok=True)

        dir_ = settings.detectors[detector].dir
        tag = dir_.lower()
        detector_args = settings.detectors[detector].get('args', '')

        cmd = f"docker run -v {save_folder}:/root/outputs -v {args.input}:{args.input} --rm --gpus all {tag} -i {args.input} {detector_args}"
        logger.log(logging.INFO, cmd)
        run_command(cmd)


    # AGGREGATE
    logger.log(logging.INFO, "Start aggregating ...")
    dataset = SPLIT_CHAR.join(args.input.split('/'))
    path = f"{cwd}/temp/{dataset}"

    # read logs for all detections for current dataset
    det_logs = glob.glob(f"{path}/*/*.txt")
    det_logs = list(map(read_file, det_logs))

    # collect detections for each image in each log
    detections_dicts: List[dict] = list(map(collect_detections, det_logs))

    # merge all detectors outputs based on image
    detections_dict_merged = dict()
    for detections_dict in detections_dicts:
        for img, (dets, lndms) in detections_dict.items():
            if img in detections_dict_merged:
                detections_dict_merged[img][0].extend(dets)
                detections_dict_merged[img][1].extend(lndms)
            else:
                detections_dict_merged[img] = [dets, lndms]

    # aggregate detections with voting (nms-like) algorithm.
    detections_dict_aggregated = dict()
    for img, (dets, lndms) in detections_dict_merged.items():
        detections_dict_aggregated[img] = aggregate_detections(dets, lndms, settings.thresh_iou, settings.min_votes)


    # SAVE
    logger.log(logging.INFO, f"Start saving in <{args.save_format}> format ...")

    save_filename = args.filename or list(filter(len, args.input.split('/')))[-1]
    os.makedirs(args.output_path, exist_ok=True)
    save_path = os.path.join(args.output_path, save_filename)

    if args.save_format == 'dataset':
        data = defaultdict(list)
        for ipath, (bboxes, kpss) in detections_dict_aggregated.items():
            for d, l in zip(bboxes, kpss):
                item = {"box": list(d[:4]), "score": float(d[4]), "landmarks": list(l)}
                ipath = filter_prefix(ipath, args.prefix)
                data[ipath].append(item)

        if '.json' not in save_path:
            save_path += '.json'
        dump_json(save_path, data)

    else:
        data = []
        for ipath, (bboxes, kpss) in detections_dict_aggregated.items():
            ipath = filter_prefix(ipath, args.prefix)
            line = [ipath, str(len(bboxes)), '$d']
            for i in range(len(bboxes)):
                conf = bboxes[i][-1]
                bbox = bboxes[i][:-1]
                bbox = list(map(int, bbox))
                bbox = list(map(str, bbox))

                landmarks = np.array(kpss[i]).astype(int).flatten()
                landmarks = list(map(str, landmarks))
                line.append(str(conf))
                line.extend(bbox)
                line.extend(landmarks)

            data.append(' '.join(line))

        if '.txt' not in save_path:
            save_path += '.txt'

        with open(save_path, 'w') as f:
            f.write('\n'.join(data))


    # CLEAN UP
    if not args.keep_temp:
        shutil.rmtree(f"{cwd}/temp")
