#include <array>
#include <cmath>
#include <cstring> // for memcpy
#include <vector>
#include <x86intrin.h> // Required for SIMD instructions on x86 architectures


constexpr size_t _m256_float_step_sz = sizeof(__m256) / sizeof(float);


float l2_norm(float* const arr){
	__m256 product = _mm256_set1_ps(0.0);
	for (unsigned i = 0;  i < 1024/8;  ++i){
		__m256 row = _mm256_loadu_ps(arr + 8*i);
		product += _mm256_mul_ps(row, row);
	}
	return std::sqrt(product[0] + product[1] + product[2] + product[3]);
}
void normalize(float* const flat_arr){
	const __m256 norm = _mm256_set1_ps(l2_norm(flat_arr)); // NOTE: Requires "-march=native" otherwise you get the error: "inlining failed in call to always_inline '__m256 _mm256_set1_ps(float)': target specific option mismatch"
	// this takes the float value from l2_norm, creates a __m256 (aka 8 floats) set to that value
	for (std::size_t i = 0;  i < 1024/8;  ++i){
		__m256 hwvec2 = _mm256_load_ps(&flat_arr[i * _m256_float_step_sz]);
		_mm256_store_ps(&flat_arr[i * _m256_float_step_sz], _mm256_div_ps(hwvec2, norm));
	}
}

void cosine_similarity(float* const vecarr,  float* const result,  const std::size_t num_rows){
	for (unsigned i = 0;  i < num_rows;  ++i){
		normalize(vecarr + 1024*i);
	}
	for (int i = 0;  i < num_rows;  ++i){
		for (int j = 0;  j < num_rows;  ++j){
			__m256 product = _mm256_set1_ps(0.0);
			for (int k = 0;  k < 1024/8;  ++k){
				__m256 row1 = _mm256_load_ps(&vecarr[1024*i + 8*k]);
				__m256 row2 = _mm256_load_ps(&vecarr[1024*j + 8*k]);
				__m256 product2 = _mm256_mul_ps(row1, row2);
				product = _mm256_add_ps(product, product2);
			}
			result[num_rows*j + i] = product[0] + product[1] + product[2] + product[3];
		}
		// speed is 200 per 1s
		if (i%200 == 0)
			printf("DONE %i\n", i);
	}
}


void* aligned_malloc(const std::size_t required_bytes,  const std::size_t alignment){
	void* orig_ptr;
	void** p2; // aligned block
	int offset = alignment - 1 + sizeof(void*);
	if ((orig_ptr = (void*)malloc(required_bytes + offset)) == nullptr){
		return nullptr;
	}
	p2 = (void**)(((size_t)(orig_ptr) + offset) & ~(alignment - 1));
	p2[-1] = orig_ptr;
	return p2;
}

extern "C"
float* cosine_similarity_from_numpy_contiguous_array(float* const float_array_ptr,  const std::size_t n_elements){
	float* const vecarr = reinterpret_cast<float*>(aligned_malloc(n_elements*1024*sizeof(float), 32));
	for (size_t i = 0;  i < n_elements;  ++i){
		memcpy(vecarr + 1024*i,  float_array_ptr + i * 1024,  1024*sizeof(float));
	}
	float* const result = reinterpret_cast<float*>(aligned_malloc(n_elements*n_elements*sizeof(float), 32));
	//memset(result, 0, n_elements*n_elements*sizeof(float));
	cosine_similarity(vecarr, result, n_elements);
	void* vecarr_orig_ptr = &vecarr[-1];
	free(vecarr_orig_ptr);
	return result;
}
