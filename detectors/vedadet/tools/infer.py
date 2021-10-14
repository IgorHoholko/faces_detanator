import argparse
import os
import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

from vedacore.image import imread, imwrite
from vedacore.misc import Config, color_val, load_weights
from vedacore.parallel import collate, scatter
from vedadet.datasets.pipelines import Compose
from vedadet.engines import build_engine





def prepare(cfg):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'
    engine = build_engine(cfg.infer_engine)

    engine.model.to(device)
    load_weights(engine.model, cfg.weights.filepath)

    data_pipeline = Compose(cfg.data_pipeline)
    return engine, data_pipeline, device


def plot_result(result, imgfp, class_names, outfp='out.jpg'):
    font_scale = 0.5
    bbox_color = 'green'
    text_color = 'green'
    thickness = 1

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    img = imread(imgfp)

    bboxes = np.vstack(result)
    labels = [
        np.full(bbox.shape[0], idx, dtype=np.int32)
        for idx, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox[:4].astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness)
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    imwrite(img, outfp)


def _parse_args():
    parser = argparse.ArgumentParser(description='Infer a detector')
    add = parser.add_argument
    add('-c', '--config', help='config file path')
    add('-i', '--input', type=str)
    add('-t', '--thresh', type=float, default=0.5)
    add('-s', '--save_path', type=str, default='./')
    add('--draw', action='store_true', default=False)

    args = parser.parse_args()
    return args

def main():

    args = _parse_args()
    cfg = Config.fromfile(args.config)
    cfg.infer_engine['test_cfg']['score_thr'] = args.thresh

    print(f"Input: {args.input}")

    class_names = cfg.class_names

    engine, data_pipeline, device = prepare(cfg)


    if args.input.split('.')[-1] in ('jpg', 'png'):
        img_paths = [args.input]
    else:
        img_paths = glob.glob(f"{args.input}/**/*.jpg", recursive=True)
        img_paths.extend(  glob.glob(f"{args.input}/**/*.png", recursive=True) )

    os.makedirs(args.save_path, exist_ok=True)

    output = dict()

    for imgname in tqdm(img_paths):

        data = dict(img_info=dict(filename=imgname), img_prefix=None)

        data = data_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if device != 'cpu':
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            # just get the actual data from DataContainer
            data['img_metas'] = data['img_metas'][0].data
            data['img'] = data['img'][0].data
        result = engine.infer(data['img'], data['img_metas'])[0]

        if args.draw:
            plot_result(result, imgname, class_names)

        output[imgname] = result[0]

    data = []
    for ipath, bboxes in output.items():
        line = [ipath, str(len(bboxes)), '$d']
        for i in range(len(bboxes)):

            conf = bboxes[i][-1]
            bbox = bboxes[i][:-1]
            bbox = list(map(int, bbox))
            bbox = list(map(str, bbox))

            landmarks = np.zeros(10) - 1
            landmarks = landmarks.astype(int)
            landmarks = list(map(str, landmarks))
            line.append(str(conf))
            line.extend(bbox)
            line.extend(landmarks)

        data.append(' '.join(line))

    with open(os.path.join(args.save_path, 'meta.txt'), 'w') as f:
        f.write('\n'.join(data))




if __name__ == '__main__':
    main()
