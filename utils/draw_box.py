from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from typing import List


class ColorMap(object):
    def __init__(self, num):
        super().__init__()
        self.get_color_map_list(num)
        self.color_map = {}
        self.ptr = 0

    def __getitem__(self, key):
        return self.color_map[key]

    def update(self, keys):
        for key in keys:
            if key not in self.color_map:
                i = self.ptr % len(self.color_list)
                self.color_map[key] = self.color_list[i]
                self.ptr += 1

    def get_color_map_list(self, num_classes):
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
                color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
                color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
                j += 1
                lab >>= 3
        self.color_list = [
            color_map[i:i + 3] for i in range(0, len(color_map), 3)
        ]




class BoxesDrawer:
    def __init__(self):
        self.color_map = ColorMap(100)
        self.font_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "SourceHanSansCN-Medium.otf")

    def draw(self, img: np.array, box_list: List[np.array], confidences: List[float], labels: List[int], landmarks = None):
        self.color_map.update(labels)
        im = Image.fromarray(img)
        draw = ImageDraw.Draw(im)

        for i in range(len(box_list)):
            bbox = box_list[i]
            score = confidences[i]
            label = labels[i]
            lndms = landmarks[i] if landmarks else None

            color = tuple(self.color_map[label])

            xmin, ymin, xmax, ymax = bbox

            font_size = max(int((xmax - xmin) // 6), 10)
            font = ImageFont.truetype(self.font_path, font_size)

            text = "{} {:.4f}".format(label, score)
            th = sum(font.getmetrics())
            tw = font.getsize(text)[0]
            start_y = max(0, ymin - th)


            draw.rectangle(
                [(xmin, start_y), (xmin + tw + 1, start_y + th)], fill=color)
            draw.text(
                (xmin + 1, start_y),
                text,
                fill=(255, 255, 255),
                font=font,
                anchor="la")
            draw.rectangle(
                [(xmin, ymin), (xmax, ymax)], width=2, outline=color)
            # plt.imshow(img); plt.scatter( lndms[::2], lndms[1::2], c='r' ); plt.show()
            if lndms:
                for x, y in np.reshape(lndms, (-1, 2)):
                    r = (xmax - xmin) // 10
                    draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
        return np.array(im)