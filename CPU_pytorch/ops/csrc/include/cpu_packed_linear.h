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

template <typename T>
class CPU_Packed_Linear
{
public:
    CPU_Packed_Linear(int in_features, int out_features, bool bias);
    ~CPU_Packed_Linear();

    void pack_tensor(torch::Tensor t);
    T *data_ptr() { return _packedB; }
    bool is_Packed() { return _isPacked; }
    const int _in_features;
    const int _out_features;

private:
    bool _bias;
    int _newN, _newK;
    bool _isPacked;
    T *_packedB;
    void _free_internal_ptr()
    {
        if (_isPacked)
            ::operator delete[](_packedB, addr_align);
        _isPacked = false;
    }
};

template <typename T>
CPU_Packed_Linear<T>::CPU_Packed_Linear(int in_features, int out_features, bool bias) : _in_features(in_features), _out_features(out_features)
{
    size_t M = 0, N, K; // Here we don't actually need M, so we assign a dummy value to it.
    auto newSize = get_packed_size(M, out_features, in_features);
    _newN = newSize[1];
    _newK = newSize[2];
    _isPacked = false;
}

template <typename T>
CPU_Packed_Linear<T>::~CPU_Packed_Linear()
{
    _free_internal_ptr();
}

template <typename T>
void CPU_Packed_Linear<T>::pack_tensor(torch::Tensor t)
{
    _free_internal_ptr();
    size_t total_size = _newN * _newK;
    _packedB = packing(_out_features, _in_features, (float *)t.t().contiguous().data_ptr(), t.size(0));
    _isPacked = true;
}