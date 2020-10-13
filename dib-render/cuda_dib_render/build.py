from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='dr_batch_dib_render',
    ext_modules=[
        CUDAExtension('dr_batch_dib_render', [
            'dr.cpp',
            'dr_cuda_for.cu',
            'dr_cuda_back.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
