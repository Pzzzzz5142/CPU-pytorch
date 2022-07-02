//
// Created by Pzzzzz on 2022/7/1.
//
#pragma once

#define NOMINMAX // Windows idiosyncrasy
                 // https://stackoverflow.com/questions/4913922/possible-problems-with-nominmax-on-visual-c

#include <stdio.h>
#include <cassert>
#include <algorithm>
#include <cmath>
#include "simd.h"

class CPU_Linear
{
public:
    CPU_Linear(int in_features, int out_features, bool bias) : _in_features(in_features), _out_features(out_features), _bias(bias) {}
    ~CPU_Linear() {}

private:
    int _in_features;
    int _out_features;
    bool _bias;
};