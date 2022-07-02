from builder import TorchCPUOpBuilder


class CPULinearBuilder(TorchCPUOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_LINEAR"
    NAME = "cpu_linear"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f"deepspeed.ops.linear.{self.NAME}_op"

    def sources(self):
        return ["csrc/linear/cpu_linear.cpp"]

    def libraries_args(self):
        args = super().libraries_args()
        args += ["/opt/intel/oneapi/mkl/lastest/lib"]
        return args

    def include_paths(self):
        return ["csrc/include", "/opt/intel/oneapi/mkl/latest/include"]
