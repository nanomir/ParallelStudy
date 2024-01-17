#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <omp.h>
#include "thread_misc.h"

struct profiling_result_t {
	double result = 0, time = 0, speedup = 0, efficiency = 0;
	unsigned T = 1;
};

struct measure_func {
	std::string name;
	double (*func)(const uint32_t*, size_t);
	measure_func(std::string name, double (*func)(const uint32_t*, size_t)) : name(name), func(func)
	{
	}
};

std::vector<profiling_result_t> run_experiement_omp(double(*f)(const uint32_t*, size_t), size_t N, uint32_t* arr);
std::vector<profiling_result_t> run_experiement_cpp(double (*f)(const uint32_t*, size_t), size_t N, uint32_t* arr);

template <class T, std::unsigned_integral V>
T fast_pow(T x, V n) requires requires(T x) { T(1); x *= x; } {
	T r = T(1);

	while (n > 0) {
		if (n & 1) {
			r *= x;
		}
		x *= x;
		n >>= 1;
	}

	return r;
}