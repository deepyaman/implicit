import glob
import io
import logging
import os.path
import platform
import sys

from setuptools import Extension, find_packages, setup

# Adapted from https://github.com/rmcgibbo/npcuda-example and
# https://github.com/cupy/cupy/blob/master/cupy_setup_build.py
import logging
import os
import sys
from distutils import ccompiler, errors, msvccompiler, unixccompiler
from distutils.spawn import find_executable

from setuptools.command.build_ext import build_ext as setuptools_build_ext


def locate_cuda():
    """Locate the CUDA environment on the system

    If a valid cuda installation is found this returns a dict with keys 'home', 'nvcc', 'include',
    and 'lib64' and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything is based on finding
    'nvcc' in the PATH.

    If nvcc can't be found, this returns None
    """
    nvcc_bin = "nvcc"
    if sys.platform.startswith("win"):
        nvcc_bin = "nvcc.exe"

    # first check if the CUDAHOME env variable is in use
    nvcc = find_executable(nvcc_bin)

    home = None
    if "CUDAHOME" in os.environ:
        home = os.environ["CUDAHOME"]
    elif "CUDA_PATH" in os.environ:
        home = os.environ["CUDA_PATH"]

    if not nvcc or not os.path.exists(nvcc):
        # if we can't find nvcc or it doesn't exist, try getting from root cuda directory
        nvcc = os.path.join(home, "bin", nvcc_bin) if home else None
        if not nvcc or not os.path.exists(nvcc):
            logging.warning(
                "The nvcc binary could not be located in your $PATH. Either add it to "
                "your path, or set $CUDAHOME to enable CUDA extensions"
            )
            return None

    if not home:
        home = os.path.dirname(os.path.dirname(nvcc))

    if not os.path.exists(os.path.join(home, "include")) or not os.path.exists(
        os.path.join(home, "lib64")
    ):
        logging.warning("Failed to find cuda include directory, attempting /usr/local/cuda")
        home = "/usr/local/cuda"

    cudaconfig = {
        "home": home,
        "nvcc": nvcc,
        "include": os.path.join(home, "include"),
        "lib64": os.path.join(home, "lib64"),
    }

    post_args = [
        "-arch=sm_60",
        "-gencode=arch=compute_50,code=sm_50",
        "-gencode=arch=compute_52,code=sm_52",
        "-gencode=arch=compute_60,code=sm_60",
        "-gencode=arch=compute_61,code=sm_61",
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_70,code=compute_70",
        "--ptxas-options=-v",
        "--extended-lambda",
        "-O2",
    ]

    if sys.platform == "win32":
        cudaconfig["lib64"] = os.path.join(home, "lib", "x64")
        post_args += ["-Xcompiler", "/MD"]
    else:
        post_args += ["-c", "--compiler-options", "'-fPIC'"]

    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            logging.warning("The CUDA %s path could not be located in %s", k, v)
            return None

    cudaconfig["post_args"] = post_args
    return cudaconfig


# This code to build .cu extensions with nvcc is taken from cupy:
# https://github.com/cupy/cupy/blob/master/cupy_setup_build.py
class _UnixCCompiler(unixccompiler.UnixCCompiler):
    src_extensions = list(unixccompiler.UnixCCompiler.src_extensions)
    src_extensions.append(".cu")

    def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        # For sources other than CUDA C ones, just call the super class method.
        if os.path.splitext(src)[1] != ".cu":
            return unixccompiler.UnixCCompiler._compile(
                self, obj, src, ext, cc_args, extra_postargs, pp_opts
            )

        # For CUDA C source files, compile them with NVCC.
        _compiler_so = self.compiler_so
        try:
            nvcc_path = CUDA["nvcc"]
            post_args = CUDA["post_args"]
            # TODO? base_opts = build.get_compiler_base_options()
            self.set_executable("compiler_so", nvcc_path)

            return unixccompiler.UnixCCompiler._compile(
                self, obj, src, ext, cc_args, post_args, pp_opts
            )
        finally:
            self.compiler_so = _compiler_so


class _MSVCCompiler(msvccompiler.MSVCCompiler):
    _cu_extensions = [".cu"]

    src_extensions = list(unixccompiler.UnixCCompiler.src_extensions)
    src_extensions.extend(_cu_extensions)

    def _compile_cu(
        self,
        sources,
        output_dir=None,
        macros=None,
        include_dirs=None,
        debug=0,
        extra_preargs=None,
        extra_postargs=None,
        depends=None,
    ):
        # Compile CUDA C files, mainly derived from UnixCCompiler._compile().
        macros, objects, extra_postargs, pp_opts, _build = self._setup_compile(
            output_dir, macros, include_dirs, sources, depends, extra_postargs
        )

        compiler_so = CUDA["nvcc"]
        cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
        post_args = CUDA["post_args"]

        for obj in objects:
            try:
                src, _ = _build[obj]
            except KeyError:
                continue
            try:
                self.spawn([compiler_so] + cc_args + [src, "-o", obj] + post_args)
            except errors.DistutilsExecError as e:
                raise errors.CompileError(str(e))

        return objects

    def compile(self, sources, **kwargs):
        # Split CUDA C sources and others.
        cu_sources = []
        other_sources = []
        for source in sources:
            if os.path.splitext(source)[1] == ".cu":
                cu_sources.append(source)
            else:
                other_sources.append(source)

        # Compile source files other than CUDA C ones.
        other_objects = msvccompiler.MSVCCompiler.compile(self, other_sources, **kwargs)

        # Compile CUDA C sources.
        cu_objects = self._compile_cu(cu_sources, **kwargs)

        # Return compiled object filenames.
        return other_objects + cu_objects


class cuda_build_ext(setuptools_build_ext):
    """Custom `build_ext` command to include CUDA C source files."""

    def run(self):
        if CUDA is not None:

            def wrap_new_compiler(func):
                def _wrap_new_compiler(*args, **kwargs):
                    try:
                        return func(*args, **kwargs)
                    except errors.DistutilsPlatformError:
                        if not sys.platform == "win32":
                            CCompiler = _UnixCCompiler
                        else:
                            CCompiler = _MSVCCompiler
                        return CCompiler(None, kwargs["dry_run"], kwargs["force"])

                return _wrap_new_compiler

            ccompiler.new_compiler = wrap_new_compiler(ccompiler.new_compiler)
            # Intentionally causes DistutilsPlatformError in
            # ccompiler.new_compiler() function to hook.
            self.compiler = "nvidia"

        setuptools_build_ext.run(self)


CUDA = locate_cuda()
build_ext = cuda_build_ext if CUDA else setuptools_build_ext

NAME = "implicit"
VERSION = "0.4.8"


use_openmp = True


def define_extensions():
    if sys.platform.startswith("win"):
        # compile args from
        # https://msdn.microsoft.com/en-us/library/fwkeyyhe.aspx
        compile_args = ["/O2", "/openmp"]
        link_args = []
    else:
        gcc = extract_gcc_binaries()
        if gcc is not None:
            rpath = "/usr/local/opt/gcc/lib/gcc/" + gcc[-1] + "/"
            link_args = ["-Wl,-rpath," + rpath]
        else:
            link_args = []

        compile_args = ["-Wno-unused-function", "-Wno-maybe-uninitialized", "-O3", "-ffast-math"]
        if use_openmp:
            compile_args.append("-fopenmp")
            link_args.append("-fopenmp")

        compile_args.append("-std=c++11")
        link_args.append("-std=c++11")

    # we need numpy to build so we can include the arrayobject.h in the .cpp builds
    # try:
    #     import numpy as np
    # except ImportError:
    #     raise ValueError("numpy is required to build from source")

    src_ext = ".pyx"
    modules = [
        Extension(
            "implicit." + name,
            [os.path.join("implicit", name + src_ext)],
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
        for name in ["_nearest_neighbours", "lmf", "evaluation"]
    ]
    modules.extend(
        [
            Extension(
                "implicit.cpu." + name,
                [os.path.join("implicit", "cpu", name + src_ext)],
                language="c++",
                extra_compile_args=compile_args,
                extra_link_args=link_args,
            )
            for name in ["_als", "bpr"]
        ]
    )
    modules.append(
        Extension(
            "implicit." + "recommender_base",
            [
                os.path.join("implicit", "recommender_base" + src_ext),
                os.path.join("implicit", "topnc.cpp"),
            ],
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
    )

    if CUDA:
        conda_prefix = os.getenv("CONDA_PREFIX")
        include_dirs = [CUDA["include"], "."]
        library_dirs = [CUDA["lib64"]]
        if conda_prefix:
            include_dirs.append(os.path.join(conda_prefix, "include"))
            library_dirs.append(os.path.join(conda_prefix, "lib"))

        modules.append(
            Extension(
                "implicit.gpu._cuda",
                [
                    os.path.join("implicit", "gpu", "_cuda" + src_ext),
                    os.path.join("implicit", "gpu", "als.cu"),
                    os.path.join("implicit", "gpu", "bpr.cu"),
                    os.path.join("implicit", "gpu", "matrix.cu"),
                    os.path.join("implicit", "gpu", "device_buffer.cu"),
                    os.path.join("implicit", "gpu", "random.cu"),
                    os.path.join("implicit", "gpu", "knn.cu"),
                ],
                language="c++",
                extra_compile_args=compile_args,
                extra_link_args=link_args,
                library_dirs=library_dirs,
                libraries=["cudart", "cublas", "curand"],
                include_dirs=include_dirs,
            )
        )
    else:
        print("Failed to find CUDA toolkit. Building without GPU acceleration.")

    try:
        from Cython.Build import cythonize

        return cythonize(modules)
    except ImportError:
        return modules


# set_gcc copied from glove-python project
# https://github.com/maciejkula/glove-python


def extract_gcc_binaries():
    """Try to find GCC on OSX for OpenMP support."""
    patterns = [
        "/opt/local/bin/g++-mp-[0-9]*.[0-9]*",
        "/opt/local/bin/g++-mp-[0-9]*",
        "/usr/local/bin/g++-[0-9]*.[0-9]*",
        "/usr/local/bin/g++-[0-9]*",
    ]
    if platform.system() == "Darwin":
        gcc_binaries = []
        for pattern in patterns:
            gcc_binaries += glob.glob(pattern)
        gcc_binaries.sort()
        if gcc_binaries:
            _, gcc = os.path.split(gcc_binaries[-1])
            return gcc
        else:
            return None
    else:
        return None


def set_gcc():
    """Try to use GCC on OSX for OpenMP support."""
    # For macports and homebrew
    if platform.system() == "Darwin":
        gcc = extract_gcc_binaries()

        if gcc is not None:
            os.environ["CC"] = gcc
            os.environ["CXX"] = gcc

        else:
            global use_openmp
            use_openmp = False
            logging.warning(
                "No GCC available. Install gcc from Homebrew " "using brew install gcc."
            )


set_gcc()


def read(file_name):
    """Read a text file and return the content as a string."""
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    with io.open(file_path, encoding="utf-8") as f:
        return f.read()


setup(
    name=NAME,
    version=VERSION,
    description="Collaborative Filtering for Implicit Feedback Datasets",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="http://github.com/benfred/implicit/",
    author="Ben Frederickson",
    author_email="ben@benfrederickson.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="Matrix Factorization, Implicit Alternating Least Squares, "
    "Collaborative Filtering, Recommender Systems",
    packages=find_packages(),
    install_requires=["numpy", "scipy>=0.16", "tqdm>=4.27"],
    setup_requires=["Cython>=0.24", "scipy>=0.16"],
    ext_modules=define_extensions(),
    cmdclass={"build_ext": build_ext},
    test_suite="tests",
)
