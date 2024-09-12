#pragma once

#ifdef __AVX2__
# include <x86intrin.h> // Required for SIMD instructions on x86 architectures
#endif
#include <cstddef> // for std::size_t
#include <cstdlib> // for malloc


namespace compsky {
namespace similarity_encoding_cnn {


#ifdef __AVX2__
float l2_norm(const float* const arr){
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
		__m256 hwvec2 = _mm256_load_ps(&flat_arr[i * 8]);
		_mm256_store_ps(&flat_arr[i * 8], _mm256_div_ps(hwvec2, norm));
	}
}
float get_result_podfsdopjdsf(const float* const arr1,  const float* const arr2){
	__m256 product = _mm256_set1_ps(0.0);
	for (int k = 0;  k < 1024/8;  ++k){
		__m256 row1 = _mm256_load_ps(&arr1[8*k]);
		__m256 row2 = _mm256_load_ps(&arr2[8*k]);
		__m256 product2 = _mm256_mul_ps(row1, row2);
		product = _mm256_add_ps(product, product2);
	}
	return product[0] + product[1] + product[2] + product[3];
}
#else
float l2_norm(const float* const arr){
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
float get_result_podfsdopjdsf(const float* const arr1,  const float* const arr2){
	float x = 0.0;
	for (int k = 0;  k < 1024;  ++k) {
		x += arr1[k] * arr2[k];
	}
	return x;
}
#endif

void cosine_similarity(float* const vecarr,  float* const result,  const std::size_t num_rows){
	for (unsigned i = 0;  i < num_rows;  ++i){
		normalize(vecarr + 1024*i);
	}
	for (int i = 0;  i < num_rows;  ++i){
		for (int j = 0;  j < num_rows;  ++j){
			result[num_rows*j + i] = get_result_podfsdopjdsf(vecarr+1024*i, vecarr+1024*j);
		}
		// speed is 200 per 5s
		if (i%200 == 0)
			printf("DONE %i\n", i);
	}
}

template<unsigned N>
void get_N_closest_from_prealigned_prenormalised_arrs_to_arr_given_prealigned_buf(const float* const vecarr,  const std::size_t num_rows,  float* const arr,  unsigned* const closest_indices_arr){
	float closest_values[N] = {}; // init to 0.0
	for (int j = 0;  j < num_rows;  ++j){
		const float result = get_result_podfsdopjdsf(vecarr+1024*j, arr);
		for (unsigned i = 0;  i < N;  ++i){
			if (result > closest_values[i]){
				for (unsigned _j = N-1;  _j > i;  --_j){
					closest_values[_j] = closest_values[_j-1];
					closest_indices_arr[_j] = closest_indices_arr[_j-1];
				}
				closest_values[i] = result;
				closest_indices_arr[i] = j;
				break;
			}
		}
	}
}

bool load_from_file(const std::uint64_t file_id,  float* const float_array_ptr,  char* const filepath_buf,  char* filepath_itr){
	compsky::asciify::asciify(filepath_itr, file_id, '\0');
	compsky::os::ReadOnlyFile f(filepath_buf);
	printf("Loading %s\t%u\n", filepath_buf, (unsigned)f.is_null());
	if (f.is_null())
		return true;
	f.read_into_buf(reinterpret_cast<char*>(float_array_ptr), 1024*sizeof(float));
	return false;
}

template<unsigned N>
void get_N_closest_from_prealigned_prenormalised_arrs_to_fileid_given_prealigned_buf(const float* const vecarr,  const std::size_t num_rows,  char* const filepath_buf,  char* filepath_itr,  const std::uint64_t file_id,  unsigned(&closest_indices_arr)[N]){
	float arr[1024];
	if (likely(not load_from_file(file_id, arr, filepath_buf, filepath_itr))){
		get_N_closest_from_prealigned_prenormalised_arrs_to_arr_given_prealigned_buf<N>(vecarr, num_rows, arr, closest_indices_arr);
	}
}

void* aligned_malloc(const std::size_t required_bytes,  const std::size_t alignment){
	void* orig_ptr;
	void** p2; // aligned block
	int offset = alignment - 1 + sizeof(void*);
	if ((orig_ptr = (void*)malloc(required_bytes + offset)) == nullptr){
		return nullptr;
	}
	p2 = (void**)(((std::size_t)(orig_ptr) + offset) & ~(alignment - 1));
	p2[-1] = orig_ptr;
	return p2;
}

const float* new_aligned_normalised_float_arr_given_file_ids(const std::int64_t* file_ids,  const unsigned n_files,  char* const filepath_buf,  char* filepath_itr){
	float* const vecarr = reinterpret_cast<float*>(aligned_malloc((n_files)*1024*sizeof(float), 32));
	for (unsigned i = 0;  i < n_files;  ++i){
		load_from_file(file_ids[i], vecarr+1024*i, filepath_buf, filepath_itr);
		normalize(vecarr + 1024*i);
	}
	return vecarr;
}


}
}
