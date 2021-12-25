# faces_detanator

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7--3.9-blue?style=for-the-badge&logo=python&logoColor=white"></a>

High accurate tool for automatic faces detection with landmarks.

The library is based on public detectors with high accuracy (TinaFace, Retinaface, SCRFD, ...). All models predict detections, then voting algorithm performs aggregation.

| | | |
|:-------------------------:|:-------------------------:| :-------------------------:|
|  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="docs/0_Parade_Parade_0_901.jpg">| <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="docs/photo_2021-12-25_16-31-22.jpg">  |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="docs/photo_2021-12-25_16-31-26.jpg">|

## Prerequisites

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
5) Install **git-lfs** to pull artifacts: `git lfs install` 


## 🚀&nbsp;&nbsp;Quickstart
```yaml
# clone project
https://github.com/IgorHoholko/faces_detanator

# [OPTIONAL] create virtual enviroment
virtualenv venv --python=python3.7
source venv/bin/activate

# install requirements
pip install -r requirements.txt
```

## Annotate your images
To start annotating, run the command:
```bash
python run.py -i <path_to_your_images>
```
For more information run:
```bash
python run.py -h
```









