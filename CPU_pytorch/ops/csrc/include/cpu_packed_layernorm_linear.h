//
// Created by Pzzzzz on 2022/8/31.
//
#pragma once

#define NOMINMAX // Windows idiosyncrasy
                 // https://stackoverflow.com/questions/4913922/possible-problems-with-nominmax-on-visual-c

#include <stdio.h>
#include <cassert>
#include <cmath>
#include "sgemm.h"
#include <torch/extension.h>
#include "cpu_packed_linear.h"

template <typename T>
class CPU_Packed_Layernorm_Linear : public CPU_Packed_Linear<T>
{
public:
    CPU_Packed_Layernorm_Linear(int in_features, int out_features, bool bias);
    ~CPU_Packed_Layernorm_Linear();

private:
};

template <typename T>
CPU_Packed_Layernorm_Linear<T>::CPU_Packed_Layernorm_Linear(int in_features, int out_features, bool bias) : CPU_Packed_Linear<T>(in_features, out_features, bias)
{
    // size_t M = 0, N, K; // Here we don't actually need M, so we assign a dummy value to it.
    // auto newSize = get_packed_size(M, out_features, in_features);
    //_newN = newSize[1];
    //_newK = newSize[2];
    //_isPacked = false;
}

template <typename T>
CPU_Packed_Layernorm_Linear<T>::~CPU_Packed_Layernorm_Linear()
{
}
