#include <torch/extension.h>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include "cpu_packed_layernorm_linear.h"

static std::unordered_map<int, std::shared_ptr<void>> s_linears;

int create_linear(int packed_linear_id, int in_features, int out_features, bool bias, bool should_log = false)
{
    auto linear = std::make_shared<CPU_Packed_Layernorm_Linear<float>>(in_features, out_features, bias);

    s_linears[packed_linear_id] = linear;

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
               packed_linear_id,
               avx_type.c_str());
        printf("Config: in_features=%d, out_features=%d, bias=%d\n", in_features, out_features, (int)bias);
    }

    return 0;
}

int destroy_linear(int packed_linear_id)
{
    s_linears.erase(packed_linear_id);

    return 0;
}

torch::Tensor cpu_linear_forward(int packed_linear_id, const torch::Tensor &input, const torch::Tensor &norm_weight, const torch::Tensor &norm_bias, const torch::optional<torch::Tensor> &bias_opt = torch::nullopt)
{
    std::shared_ptr<CPU_Packed_Layernorm_Linear<float>> lin = std::static_pointer_cast<CPU_Packed_Layernorm_Linear<float>>(s_linears[packed_linear_id]);
    assert(input.size(input.dim() - 1) == lin->_in_features);
    int64_t input_col = 1;
    std::vector<int64_t> output_size(input.dim());
    for (int i = 0; i < input.dim() - 1; i++)
        input_col *= input.size(i), output_size[i] = input.size(i);

    output_size[input.dim() - 1] = lin->_out_features;

    size_t m = input_col, n = lin->_out_features, k = lin->_in_features;

    float *norm_weight_c = (float *)norm_weight.contiguous().data_ptr();
    float *norm_bias_c = (float *)norm_bias.contiguous().data_ptr();

    if (bias_opt)
    {
        auto bias = bias_opt.value();
        auto input_c = input.contiguous();
        auto output = bias.repeat({input_col, 1}).view(output_size).contiguous();

        float *input_ptr = (float *)input_c.data_ptr();
        float *output_ptr = (float *)output.data_ptr();

        gemm_layernorm_compute_sum(input_col, n, k, input_ptr, lin->data_ptr(), output_ptr, norm_weight_c, norm_bias_c, bias_opt ? 1 : 0);

        return output;
    }
    else
    {
        auto input_c = input.contiguous();
        torch::Tensor output;
        if (bias_opt)
        {
            auto bias = bias_opt.value();
            auto output = bias.repeat({input_col, 1}).view(output_size).contiguous();
        }
        else
            output = input.new_zeros({input_col, lin->_out_features}).view(output_size).contiguous();

        float *input_ptr = (float *)input_c.data_ptr();
        float *output_ptr = (float *)output.data_ptr();

        gemm_layernorm_compute_sum(input_col, n, k, input_ptr, lin->data_ptr(), output_ptr, norm_weight_c, norm_bias_c, bias_opt ? 1 : 0);

        return output;
    }
}

torch::Tensor cpu_linear_backward(const torch::Tensor &out_grad)
{
    throw;
}

void pack_weight(int packed_linear_id, const torch::Tensor &weight)
{
    std::shared_ptr<CPU_Packed_Layernorm_Linear<float>> lin = std::static_pointer_cast<CPU_Packed_Layernorm_Linear<float>>(s_linears[packed_linear_id]);
    if (weight.dim() == 2)
    {
        assert(lin->_in_features == weight.size(1));
        assert(lin->_out_features == weight.size(0));
    }
    else
    {
        assert(lin->_in_features == weight.size(0));
        assert(lin->_out_features == 1);
    }
    lin->pack_tensor(weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("linear_forward", &cpu_linear_forward, "CPU linear forward (C++)");
    m.def("linear_backward", &cpu_linear_backward, "CPU linear backward (C++)");
    m.def("pack_weight", &pack_weight, "CPU Pack Weight Matrix (C++)");
    m.def("create_linear", &create_linear, "CPU Adam (C++)");
    m.def("destroy_linear", &destroy_linear, "CPU Adam destroy (C++)");
}
