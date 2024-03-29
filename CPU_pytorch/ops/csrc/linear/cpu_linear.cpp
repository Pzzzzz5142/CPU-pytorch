#include "cpu_linear.h"
#include <omp.h>
#include <torch/extension.h>
#include "sgemm.h"
#include <mkl.h>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>

static std::unordered_map<int, std::shared_ptr<void>> s_linears;

int create_linear(int linear_id, int in_features, int out_features, bool bias, bool should_log = false)
{
    auto linear = std::make_shared<CPU_Linear>(in_features, out_features, bias);

    s_linears[linear_id] = linear;

    if (should_log)
    {
        std::string avx_type = "";
#if defined(__AVX512__)
        avx_type = "AVX512";
#else
#if defined(__AVX256__)
        avx_type = "AVX2";
#else
        avx_type = "scalar";
#endif
#endif
        printf("CPU Linear #%d is created with %s arithmetic capability.\n",
               linear_id,
               avx_type.c_str());
        printf("Config: in_features=%d, out_features=%d, bias=%d\n", in_features, out_features, (int)bias);
    }

    return 0;
}
int destroy_linear(int linear_id)
{
    s_linears.erase(linear_id);

    return 0;
}

torch::Tensor cpu_linear_forward(const torch::Tensor &input, const torch::Tensor &weight, const torch::optional<torch::Tensor> &bias_opt = torch::nullopt)
{
    int64_t input_col = 1;
    std::vector<int64_t> output_size(input.dim());
    for (int i = 0; i < input.dim() - 1; i++)
        input_col *= input.size(i), output_size[i] = input.size(i);

    output_size[input.dim() - 1] = weight.dim() == 2 ? weight.size(0) : 1;

    size_t m = input_col, n = weight.dim() == 2 ? weight.size(0) : 1, k = input.size(input.dim() - 1);

    if (weight.dim() == 2 && bias_opt)
    {
        auto bias = bias_opt.value();
        auto input_c = input.contiguous();
        auto weight_tc = weight.t().contiguous();
        auto output = bias.repeat({input_col, 1}).view(output_size).contiguous();

        float *input_ptr = (float *)input_c.data_ptr();
        float *weight_ptr = (float *)weight_tc.data_ptr();
        float *output_ptr = (float *)output.data_ptr();

        square_gemm(input_col, n, k, input_ptr, weight_ptr, output_ptr, bool(bias_opt));
        //  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, input_col, weight_c.size(0), input_c.size(input_c.dim() - 1), 1, input_ptr, input_c.size(input_c.dim() - 1), weight_ptr, weight_c.size(1), 1, output_ptr, weight_c.size(0));

        return output;
    }
    else
    {
        auto input_c = input.contiguous();
        auto weight_tc = weight.t().contiguous();
        torch::Tensor output;
        if (bias_opt)
        {
            auto bias = bias_opt.value();
            auto output = bias.repeat({input_col, 1}).view(output_size).contiguous();
        }
        else
            output = weight.new_zeros({input_col, weight.size(0)}).view(output_size).contiguous();

        float *input_ptr = (float *)input_c.data_ptr();
        float *weight_ptr = (float *)weight_tc.data_ptr();
        float *output_ptr = (float *)output.data_ptr();

        square_gemm(input_col, n, k, input_ptr, weight_ptr, output_ptr, bool(bias_opt));
        //  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, input_col, n, input_c.size(input_c.dim() - 1), 1, input_ptr, input_c.size(input_c.dim() - 1), weight_ptr, k, 1, output_ptr, n);

        return output;
    }
}

torch::Tensor cpu_linear_backward(const torch::Tensor &out_grad)
{
    throw;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("linear_forward", &cpu_linear_forward, "CPU linear forward (C++)");
    m.def("linear_backward", &cpu_linear_backward, "CPU linear backward (C++)");
    m.def("create_linear", &create_linear, "CPU Adam (C++)");
    m.def("destroy_linear", &destroy_linear, "CPU Adam destroy (C++)");
}
