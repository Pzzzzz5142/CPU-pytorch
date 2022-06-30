"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
import os
import sys
import time
import json
import importlib
from pathlib import Path
import subprocess
import shlex
import shutil
import tempfile
import distutils.ccompiler
import distutils.log
import distutils.sysconfig
from distutils.errors import CompileError, LinkError
from abc import ABC, abstractmethod

YELLOW = "\033[93m"
END = "\033[0m"
WARNING = f"{YELLOW} [WARNING] {END}"

DEFAULT_TORCH_EXTENSION_PATH = "/tmp/torch_extensions"
DEFAULT_COMPUTE_CAPABILITIES = "6.0;6.1;7.0"

try:
    import torch
except ImportError:
    print(
        f"{WARNING} unable to import torch, please install it if you want to pre-compile any deepspeed ops."
    )
else:
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])


class OpBuilder(ABC):
    _rocm_version = None
    _is_rocm_pytorch = None

    def __init__(self, name):
        self.name = name
        self.jit_mode = False

    @abstractmethod
    def absolute_name(self):
        """
        Returns absolute build path for cases where the op is pre-installed, e.g., deepspeed.ops.adam.cpu_adam
        will be installed as something like: deepspeed/ops/adam/cpu_adam.so
        """
        pass

    @abstractmethod
    def sources(self):
        """
        Returns list of source files for your op, relative to root of deepspeed package (i.e., DeepSpeed/deepspeed)
        """
        pass

    def hipify_extension(self):
        pass

    def include_paths(self):
        """
        Returns list of include paths, relative to root of deepspeed package (i.e., DeepSpeed/deepspeed)
        """
        return []

    def cxx_args(self):
        """
        Returns optional list of compiler flags to forward to the build
        """
        return []

    def is_compatible(self, verbose=True):
        """
        Check if all non-python dependencies are satisfied to build this op
        """
        return True

    def extra_ldflags(self):
        return []

    def libraries_installed(self, libraries):
        valid = False
        check_cmd = "dpkg -l"
        for lib in libraries:
            result = subprocess.Popen(
                f"dpkg -l {lib}",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
            )
            valid = valid or result.wait() == 0
        return valid

    def has_function(self, funcname, libraries, verbose=False):
        """
        Test for existence of a function within a tuple of libraries.

        This is used as a smoke test to check whether a certain library is available.
        As a test, this creates a simple C program that calls the specified function,
        and then distutils is used to compile that program and link it with the specified libraries.
        Returns True if both the compile and link are successful, False otherwise.
        """
        tempdir = None  # we create a temporary directory to hold various files
        filestderr = None  # handle to open file to which we redirect stderr
        oldstderr = None  # file descriptor for stderr
        try:
            # Echo compile and link commands that are used.
            if verbose:
                distutils.log.set_verbosity(1)

            # Create a compiler object.
            compiler = distutils.ccompiler.new_compiler(verbose=verbose)

            # Configure compiler and linker to build according to Python install.
            distutils.sysconfig.customize_compiler(compiler)

            # Create a temporary directory to hold test files.
            tempdir = tempfile.mkdtemp()

            # Define a simple C program that calls the function in question
            prog = (
                "void %s(void); int main(int argc, char** argv) { %s(); return 0; }"
                % (funcname, funcname)
            )

            # Write the test program to a file.
            filename = os.path.join(tempdir, "test.c")
            with open(filename, "w") as f:
                f.write(prog)

            # Redirect stderr file descriptor to a file to silence compile/link warnings.
            if not verbose:
                filestderr = open(os.path.join(tempdir, "stderr.txt"), "w")
                oldstderr = os.dup(sys.stderr.fileno())
                os.dup2(filestderr.fileno(), sys.stderr.fileno())

            # Workaround for behavior in distutils.ccompiler.CCompiler.object_filenames()
            # Otherwise, a local directory will be used instead of tempdir
            drive, driveless_filename = os.path.splitdrive(filename)
            root_dir = (
                driveless_filename[0] if os.path.isabs(driveless_filename) else ""
            )
            output_dir = os.path.join(drive, root_dir)

            # Attempt to compile the C program into an object file.
            cflags = shlex.split(os.environ.get("CFLAGS", ""))
            objs = compiler.compile(
                [filename],
                output_dir=output_dir,
                extra_preargs=self.strip_empty_entries(cflags),
            )

            # Attempt to link the object file into an executable.
            # Be sure to tack on any libraries that have been specified.
            ldflags = shlex.split(os.environ.get("LDFLAGS", ""))
            compiler.link_executable(
                objs,
                os.path.join(tempdir, "a.out"),
                extra_preargs=self.strip_empty_entries(ldflags),
                libraries=libraries,
            )

            # Compile and link succeeded
            return True

        except CompileError:
            return False

        except LinkError:
            return False

        except:
            return False

        finally:
            # Restore stderr file descriptor and close the stderr redirect file.
            if oldstderr is not None:
                os.dup2(oldstderr, sys.stderr.fileno())
            if filestderr is not None:
                filestderr.close()

            # Delete the temporary directory holding the test program and stderr files.
            if tempdir is not None:
                shutil.rmtree(tempdir)

    def strip_empty_entries(self, args):
        """
        Drop any empty strings from the list of compile and link flags
        """
        return [x for x in args if len(x) > 0]

    def cpu_arch(self):
        try:
            from cpuinfo import get_cpu_info
        except ImportError as e:
            cpu_info = self._backup_cpuinfo()
            if cpu_info is None:
                return "-march=native"

        try:
            cpu_info = get_cpu_info()
        except Exception as e:
            self.warning(
                f"{self.name} attempted to use `py-cpuinfo` but failed (exception type: {type(e)}, {e}), "
                "falling back to `lscpu` to get this information."
            )
            cpu_info = self._backup_cpuinfo()
            if cpu_info is None:
                return "-march=native"

        if cpu_info["arch"].startswith("PPC_"):
            # gcc does not provide -march on PowerPC, use -mcpu instead
            return "-mcpu=native"
        return "-march=native"

    def _backup_cpuinfo(self):
        # Construct cpu_info dict from lscpu that is similar to what py-cpuinfo provides
        if not self.command_exists("lscpu"):
            self.warning(
                f"{self.name} attempted to query 'lscpu' after failing to use py-cpuinfo "
                "to detect the CPU architecture. 'lscpu' does not appear to exist on "
                "your system, will fall back to use -march=native and non-vectorized execution."
            )
            return None
        result = subprocess.check_output("lscpu", shell=True)
        result = result.decode("utf-8").strip().lower()

        cpu_info = {}
        cpu_info["arch"] = None
        cpu_info["flags"] = ""
        if "genuineintel" in result or "authenticamd" in result:
            cpu_info["arch"] = "X86_64"
            if "avx512" in result:
                cpu_info["flags"] += "avx512,"
            if "avx2" in result:
                cpu_info["flags"] += "avx2"
        elif "ppc64le" in result:
            cpu_info["arch"] = "PPC_"

        return cpu_info

    def simd_width(self):
        try:
            from cpuinfo import get_cpu_info
        except ImportError as e:
            cpu_info = self._backup_cpuinfo()
            if cpu_info is None:
                return "-D__SCALAR__"

        try:
            cpu_info = get_cpu_info()
        except Exception as e:
            self.warning(
                f"{self.name} attempted to use `py-cpuinfo` but failed (exception type: {type(e)}, {e}), "
                "falling back to `lscpu` to get this information."
            )
            cpu_info = self._backup_cpuinfo()
            if cpu_info is None:
                return "-D__SCALAR__"

        if cpu_info["arch"] == "X86_64":
            if "avx512" in cpu_info["flags"]:
                return "-D__AVX512__"
            elif "avx2" in cpu_info["flags"]:
                return "-D__AVX256__"
        return "-D__SCALAR__"

    def python_requirements(self):
        """
        Override if op wants to define special dependencies, otherwise will
        take self.name and load requirements-<op-name>.txt if it exists.
        """
        path = f"requirements/requirements-{self.name}.txt"
        requirements = []
        if os.path.isfile(path):
            with open(path, "r") as fd:
                requirements = [r.strip() for r in fd.readlines()]
        return requirements

    def command_exists(self, cmd):
        if "|" in cmd:
            cmds = cmd.split("|")
        else:
            cmds = [cmd]
        valid = False
        for cmd in cmds:
            result = subprocess.Popen(f"type {cmd}", stdout=subprocess.PIPE, shell=True)
            valid = valid or result.wait() == 0

        if not valid and len(cmds) > 1:
            print(
                f"{WARNING} {self.name} requires one of the following commands '{cmds}', but it does not exist!"
            )
        elif not valid and len(cmds) == 1:
            print(
                f"{WARNING} {self.name} requires the '{cmd}' command, but it does not exist!"
            )
        return valid

    def warning(self, msg):
        print(f"{WARNING} {msg}")

    def deepspeed_src_path(self, code_path):
        if os.path.isabs(code_path):
            return code_path
        else:
            return os.path.join(Path(__file__).parent.absolute(), code_path)

    def builder(self):
        from torch.utils.cpp_extension import CppExtension

        return CppExtension(
            name=self.absolute_name(),
            sources=self.strip_empty_entries(self.sources()),
            include_dirs=self.strip_empty_entries(self.include_paths()),
            extra_compile_args={"cxx": self.strip_empty_entries(self.cxx_args())},
            extra_link_args=self.strip_empty_entries(self.extra_ldflags()),
        )

    def load(self, verbose=True):
        # from ...git_version_info import installed_ops

        # if installed_ops[self.name]:
        #     # Ensure the op we're about to load was compiled with the same
        #     # torch/cuda versions we are currently using at runtime.
        #     return importlib.import_module(self.absolute_name())
        # else:
        return self.jit_load(verbose)

    def jit_load(self, verbose=True):
        if not self.is_compatible(verbose):
            raise RuntimeError(
                f"Unable to JIT load the {self.name} op due to it not being compatible due to hardware/software issue."
            )
        try:
            import ninja
        except ImportError:
            raise RuntimeError(
                f"Unable to JIT load the {self.name} op due to ninja not being installed."
            )

        self.jit_mode = True
        from torch.utils.cpp_extension import load

        # Ensure directory exists to prevent race condition in some cases
        ext_path = os.path.join(
            os.environ.get("TORCH_EXTENSIONS_DIR", DEFAULT_TORCH_EXTENSION_PATH),
            self.name,
        )
        os.makedirs(ext_path, exist_ok=True)

        start_build = time.time()
        sources = [self.deepspeed_src_path(path) for path in self.sources()]
        extra_include_paths = [
            self.deepspeed_src_path(path) for path in self.include_paths()
        ]

        # Torch will try and apply whatever CCs are in the arch list at compile time,
        # we have already set the intended targets ourselves we know that will be
        # needed at runtime. This prevents CC collisions such as multiple __half
        # implementations. Stash arch list to reset after build.
        op_module = load(
            name=self.name,
            sources=self.strip_empty_entries(sources),
            extra_include_paths=self.strip_empty_entries(extra_include_paths),
            extra_cflags=self.strip_empty_entries(self.cxx_args()),
            extra_ldflags=self.strip_empty_entries(self.extra_ldflags()),
            verbose=verbose,
        )
        build_duration = time.time() - start_build
        if verbose:
            print(f"Time to load {self.name} op: {build_duration} seconds")

        return op_module


class TorchCPUOpBuilder(OpBuilder):
    def extra_ldflags(self):
        return []

    def cxx_args(self):
        import torch

        # if not self.is_rocm_pytorch():
        #     CUDA_LIB64 = os.path.join(torch.utils.cpp_extension.CUDA_HOME, "lib64")
        # else:
        #     CUDA_LIB64 = os.path.join(torch.utils.cpp_extension.ROCM_HOME, "lib")
        CPU_ARCH = self.cpu_arch()
        SIMD_WIDTH = self.simd_width()

        args = super().cxx_args()
        args += [
            "-g",
            CPU_ARCH,
            "-Xpreprocessor",
            "-fopenmp",
            SIMD_WIDTH,
        ]
        return args
