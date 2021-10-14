#!/bin/bash
source venv/bin/activate
python3 tools/scrfd.py -s outputs/ "$@"
