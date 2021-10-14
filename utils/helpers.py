"""
 File name   : helpers.py
 Description : description

 Date created : 24.09.2021
 Author:  Ihar Khakholka
"""

from typing import Optional
import shlex
import subprocess


def run_command(command):
    process = subprocess.Popen(shlex.split(command), shell=False, stdout=subprocess.PIPE)
    # Poll process.stdout to show stdout live
    while True:
        output = process.stdout.readline().decode()
        if process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()

def filter_prefix(path: str, prefix: Optional[str]) -> str:
    if prefix:
        path = path[path.find(prefix) + len(prefix):]  # skip prefix
        path = path if path[0] != '/' else path[1:]
    return path