#!/usr/bin/env python
import os
import pathlib
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as setuptools_command_build_ext


class CMakeMakeExtension(Extension):

    def __init__(self, name):
        super().__init__(name, sources=[])


class build_ext(setuptools_command_build_ext):

    def run(self):
        for extension in self.extensions:
            self.build_and_install_with_cmake_make(extension)
        super().run()

    def build_and_install_with_cmake_make(self, extension):
        cwd = pathlib.Path().absolute()

        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extension_dir = pathlib.Path(self.get_ext_fullpath(extension.name))
        extension_dir.mkdir(parents=True, exist_ok=True)

        args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}'.format(
                os.path.join(str(extension_dir.parent.absolute()), 'pylspack')
            ), '-DPYLSPACK_ADDITIONAL_CMAKE_CXX_FLAGS={}'.format(
                os.getenv('PYLSPACK_ADDITIONAL_CMAKE_CXX_FLAGS', '')
            )
        ]
        os.chdir(str(build_temp))
        self.spawn(['cmake', os.path.join(str(cwd), 'src')] + args)
        if not self.dry_run:
            self.spawn(['make'])

        os.chdir(str(cwd))


def read_readme(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='pylspack',
    version='0.1.0',
    description='Python package for leverage scores computations.',
    author='Sobczyk Aleksandros',
    author_email='obc@zurich.ibm.com',
    license='MIT',
    long_description=read_readme('README.md'),
    py_modules=['pylspack.linalg_kernels', 'pylspack.leverage_scores'],
    install_requires=['scipy>=1.5.0', 'numpy>=1.19.0'],
    ext_modules=[CMakeMakeExtension(name='src')],
    cmdclass={
        'build_ext': build_ext,
    }
)
