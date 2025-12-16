import os
import re
import sys
import platform
import subprocess
import torch
import pybind11 

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from packaging.version import Version, parse

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                                ", ".join(e.name for e in self.extensions))
        if platform.system() == "Windows":
            cmake_version_str = re.search(r'version\s*([\d.]+)', out.decode()).group(1)
            cmake_version = parse(cmake_version_str)
            if cmake_version < Version('3.1.0'):
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        cmake_prefix_path = f"{torch.utils.cmake_prefix_path};{pybind11.get_cmake_dir()}"

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DPython_EXECUTABLE=' + sys.executable, 
            '-DCMAKE_BUILD_TYPE=' + ('Debug' if self.debug else 'Release'),
            '-DCMAKE_PREFIX_PATH=' + cmake_prefix_path, 
            '-DEXAMPLE_VERSION_INFO=' + self.distribution.get_version()
        ]

        build_args = ['--config', 'Release']

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format('Release', extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.', '--target', 'lm_decoder'] + build_args, cwd=self.build_temp)

setup(
    name='lm_decoder',
    version='0.0.1',
    author='Wenetspeech',
    author_email='wenetspeech@wenetspeech.com',
    description='Wenet lm decoder',
    long_description='',
    ext_modules=[CMakeExtension('lm_decoder')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)