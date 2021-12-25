# faces_detanator

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7--3.9-blue?style=for-the-badge&logo=python&logoColor=white"></a>

High accurate tool for automatic faces detection with landmarks.

The library is based on public detectors with high accuracy (TinaFace, Retinaface, SCRFD, ...) which are combined together to form an ansamle. All models predict detections, then voting algorithm performs aggregation.

| | | |
|:-------------------------:|:-------------------------:| :-------------------------:|
|  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="docs/0_Parade_Parade_0_901.jpg">| <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="docs/photo_2021-12-25_16-31-22.jpg">  |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="docs/photo_2021-12-25_16-31-26.jpg">|

## :hammer_and_wrench:Prerequisites

1) [Install Docker](https://docs.docker.com/engine/install/ubuntu/)
2) [Install Nvidia Docker Container Runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
3) Install **nvidia-container-runtime**: `apt-get install nvidia-container-runtime`
4) Set `"default-runtime" : "nvidia"` in `/etc/docker/daemon.json`:
    ```json
    {
        "default-runtime": "nvidia",
        "runtimes": {
            "nvidia": {
                "path": "nvidia-container-runtime",
                "runtimeArgs": []
            }
        }
    }
    ```
5) Restart Docker: `systemctl restart docker`
6) Install **git-lfs** to pull artifacts: `git lfs install` 


## 🚀&nbsp;&nbsp;Quickstart
> docker can require sudo permission and it is used in `run.py` script. So in this case run `run.py` script with sudo permission or add your user to docker group.
> 
```yaml
# clone project
https://github.com/IgorHoholko/faces_detanator

# [OPTIONAL] create virtual enviroment
virtualenv venv --python=python3.7
source venv/bin/activate

# install requirements
pip install -r requirements.txt
```

## :boom: Annotate your images
To start annotating, run the command:
```bash
python run.py -i <path_to_your_images>
```
For more information run:
```bash
python run.py -h
```

## :scream: More functions?
You can visualize your results:
```bash
python -m helpers.draw_output -i <your_meta> -h
```

You can filter your metadata by threshold after it is formed. Just run:
```bash
python -m helpers.filter_output_by_conf -i <your_meta> -t <thres> -h
```


## :eyes: Adding new detectors for ansamble

To add new detector to ansamble you need to perform the next steps:

> Take a look at existing detectors to make process easier.

1) Create a folder for your detector <**detector**> in `detectors/` folder.
2) Prepare inference script for your detector. First, define `"-i", "--input"` argparse parameter which is responsible for input. The script to process the input:
```python
if args.input.split('.')[-1] in ('jpg', 'png'):
    img_paths = [args.input]
else:
    img_paths = glob.glob(f"{args.input}/**/*.jpg", recursive=True)
    img_paths.extend(  glob.glob(f"{args.input}/**/*.png", recursive=True) )
```
3) Next create `"-o", "--output"` argparse parameter. The place where annotation will be saved
4) Now you need to save your annotations in required format. The script to save annotations looks like this:
```python
data = []
for ipath, (bboxes, kpss) in output.items():
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

with open(os.path.join(args.output, 'meta.txt'), 'w') as f:
    f.write('\n'.join(data))
```
> If your detector doesn't provide landmarks - set landmarks to be array with all -1
5) When inference script is ready, create **entrypoint.sh** in the root of <**detector**> folder.  **entrypoint.sh** describes the logic how to infer your detector. It can look like this:
```bash
#!/bin/bash
source venv/bin/activate
python3 tools/scrfd.py -s outputs/ "$@"
```
> **IMPORTANT** set `-s` here to *outputs*.
6) Now create Dockerfile for your detector with defined earlier entrypoint.
7) Add your detector to settings.yaml by the sample.
8) Done!




