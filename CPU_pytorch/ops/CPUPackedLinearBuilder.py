from .builder import TorchCPUOpBuilder


class CPUPackedLinearBuilder(TorchCPUOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_PACKED_LINEAR"
    NAME = "cpu_packed_linear"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f"deepspeed.ops.linear.{self.NAME}_op"

    def sources(self):
        return ["csrc/linear/cpu_packed_linear.cpp"]

    def libraries_args(self):
        args = super().libraries_args()
        try:
            import os

            paths += [os.path.join(os.environ["CONDA_PREFIX"], "lib")]
        except:
            ...
        return args

    def include_paths(self):
        paths = ["csrc/include"]
        try:
            import os

            paths += [os.path.join(os.environ["CONDA_PREFIX"], "include")]
        except:
            ...
        return paths
