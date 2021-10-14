#!/usr/bin/env python
from setuptools import find_packages, setup

import os
import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def make_cuda_ext(name, module, sources, sources_cuda=[]):

    define_macros = []
    extra_compile_args = {'cxx': []}

    # if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
    define_macros += [('WITH_CUDA', None)]
    extension = CUDAExtension
    extra_compile_args['nvcc'] = [
        '-D__CUDA_NO_HALF_OPERATORS__',
        '-D__CUDA_NO_HALF_CONVERSIONS__',
        '-D__CUDA_NO_HALF2_OPERATORS__',
    ]
    sources += sources_cuda

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


if __name__ == '__main__':
    setup(
        name='vedadet',
        version='0.1.0',
        description='Single Stage Detector Toolbox',
        url='https://github.com/Media-Smart/vedadet',
        author='Yichao Xiong',
        author_email='xyc_sjtu@163.com',
        classifiers=[
            'Programming Language :: Python :: 3',
            'Operating System :: Linux',
            'License :: OSI Approved :: Apache Software License',
        ],
        keywords='computer vision, single stage, object detection',
        packages=find_packages(include=('vedacore', 'vedadet')),
        package_data={'vedadet.ops': ['*/*.so']},
        setup_requires=parse_requirements('requirements/build.txt'),
        # tests_require=parse_requirements('requirements.txt'),
        install_requires=parse_requirements('requirements/runtime.txt'),
        # extras_require={
        #     'tests': parse_requirements('requirements/tests.txt'),
        # },
        ext_modules=[
            make_cuda_ext(
                name='nms_ext',
                module='vedadet.ops.nms',
                sources=['src/nms_ext.cpp', 'src/cpu/nms_cpu.cpp'],
                sources_cuda=[
                    'src/cuda/nms_cuda.cpp', 'src/cuda/nms_kernel.cu'
                ]),
            make_cuda_ext(
                name='sigmoid_focal_loss_ext',
                module='vedadet.ops.sigmoid_focal_loss',
                sources=['src/sigmoid_focal_loss_ext.cpp'],
                sources_cuda=['src/cuda/sigmoid_focal_loss_cuda.cu']),
            make_cuda_ext(
                name='deform_conv_ext',
                module='vedadet.ops.dcn',
                sources=['src/deform_conv_ext.cpp'],
                sources_cuda=[
                    'src/cuda/deform_conv_cuda.cpp',
                    'src/cuda/deform_conv_cuda_kernel.cu'
                ]),
            make_cuda_ext(
                name='deform_pool_ext',
                module='vedadet.ops.dcn',
                sources=['src/deform_pool_ext.cpp'],
                sources_cuda=[
                    'src/cuda/deform_pool_cuda.cpp',
                    'src/cuda/deform_pool_cuda_kernel.cu'
                ])],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
