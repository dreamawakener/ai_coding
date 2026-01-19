%%writefile setup.py
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = ['src/bindings.cpp', 'src/tensor_lib.cu']

setup(
    name='mytensor',
    version='0.1',
    ext_modules=[
        CUDAExtension(
            name='mytensor',
            sources=sources,
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']},
            libraries=['cublas', 'curand']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)