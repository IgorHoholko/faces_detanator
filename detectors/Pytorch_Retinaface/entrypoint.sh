#!/bin/bash
source venv/bin/activate
CUDA_VISIBLE_DEVICES="0" python3 -m detect -s outputs/ "$@"