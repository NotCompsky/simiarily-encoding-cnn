#include <array>
#include <cmath>
#include <cstring> // for memcpy
#include <vector>
#include <x86intrin.h> // Required for SIMD instructions on x86 architectures


float l2_norm(float* const arr){
	float res = 0.0;
	for (unsigned i = 0;  i < 1024;  ++i)
		res += arr[i] * arr[i];
	return std::sqrt(res);
}
void normalize(float* const flat_arr){
	const float norm = l2_norm(flat_arr);
	for (std::size_t i = 0;  i < 1024;  ++i){
		flat_arr[i] /= norm;
	}
}

void cosine_similarity(float* const vecarr,  float* const result,  const std::size_t num_rows){
	for (unsigned i = 0;  i < num_rows;  ++i){
		normalize(vecarr + 1024*i);
	}
	for (int i = 0;  i < num_rows;  ++i){
		for (int j = 0;  j < num_rows;  ++j){
			result[num_rows*j + i] = 0;
			for (int k = 0;  k < 1024;  ++k) {
				result[num_rows*j + i] += vecarr[1024*i + k] * vecarr[1024*j + k];
			}
		}
		// speed is 200 per 5s
		if (i%200 == 0)
			printf("DONE %i\n", i);
	}
	printf("DONE cosine_similarity\n"); fflush(stdout);
}

extern "C"
float* cosine_similarity_from_numpy_contiguous_array(float* const float_array_ptr,  const std::size_t n_elements){
	alignas(16) float* const vecarr = reinterpret_cast<float*>(malloc(n_elements*1024*sizeof(float)));
	for (size_t i = 0;  i < n_elements;  ++i){
		memcpy(vecarr + 1024*i,  float_array_ptr + i * 1024,  1024*sizeof(float));
	}
	float* const result = reinterpret_cast<float*>(malloc(n_elements*n_elements*sizeof(float)));
	cosine_similarity(vecarr, result, n_elements);
	free(vecarr);
	printf("ptr == %lu\n", (uint64_t)result); fflush(stdout);
	return result;
}
