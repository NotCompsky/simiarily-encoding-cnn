#include <array>
#include <cmath>
#include <cstring> // for memcpy
#include <vector>
#include <compsky/asciify/asciify.hpp>
#include <compsky/os/read.hpp>
#include <compsky/os/write.hpp>
#include "lib.hpp"

constexpr const char* subdirs[3] = {"","cnn/","text/"};

char filepath_buf[4096] = "/path/is/not/yet/initialised/";
std::size_t path_buf_dirstart_offset = 29;

extern "C"
void set_path_buf_dir(const char* const dirpath){
	path_buf_dirstart_offset = strlen(dirpath);
	if (dirpath[path_buf_dirstart_offset-1] != '/')
		++path_buf_dirstart_offset;
	memcpy(filepath_buf, dirpath, path_buf_dirstart_offset);
	filepath_buf[path_buf_dirstart_offset-1] = '/';
}

using namespace compsky::similarity_encoding_cnn;

extern "C"
void write_to_file(const std::uint64_t file_id,  const float* const float_array_ptr,  const unsigned which_subdir){
	char* filepath_itr = filepath_buf+path_buf_dirstart_offset;
	compsky::asciify::asciify(filepath_itr, subdirs[which_subdir], file_id, '\0');
	compsky::os::WriteOnlyFile f(filepath_buf);
	f.write_from_buffer(reinterpret_cast<const char*>(float_array_ptr), 1024*sizeof(float));
}

extern "C"
int load_from_file(const std::uint64_t file_id,  float* const float_array_ptr,  const unsigned which_subdir){
	char* filepath_itr = filepath_buf+path_buf_dirstart_offset;
	compsky::asciify::asciify(filepath_itr, subdirs[which_subdir], file_id, '\0');
	compsky::os::ReadOnlyFile f(filepath_buf);
	if (f.is_null())
		return 1;
	f.read_into_buf(reinterpret_cast<char*>(float_array_ptr), 1024*sizeof(float));
	return 0;
}

extern "C"
float* cosine_similarity_from_numpy_contiguous_array(float* const float_array_ptr,  const std::size_t n_elements){
	float* const vecarr = reinterpret_cast<float*>(aligned_malloc(n_elements*1024*sizeof(float), 32));
	memcpy(vecarr,  float_array_ptr,  1024*sizeof(float)*n_elements);
	float* const result = reinterpret_cast<float*>(aligned_malloc(n_elements*n_elements*sizeof(float), 32));
	//memset(result, 0, n_elements*n_elements*sizeof(float));
	cosine_similarity(vecarr, result, n_elements);
	void* vecarr_orig_ptr = reinterpret_cast<void**>(vecarr)[-1];
	free(vecarr_orig_ptr);
	return result;
}

extern "C"
void get_10_closest_from_arrs_to_arr(float* const numpy_arrs,  const std::size_t num_rows,  float* const numpy_arr,  unsigned* const closest_indices_arr){
	float* const vecarr = reinterpret_cast<float*>(aligned_malloc((num_rows+1)*1024*sizeof(float), 32));
	memcpy(vecarr,  numpy_arrs,  1024*sizeof(float)*num_rows);
	float* const arr    = vecarr + 1024*num_rows;
	memcpy(arr,     numpy_arr,   1024*sizeof(float));
	
	for (unsigned i = 0;  i < num_rows;  ++i){
		normalize(vecarr + 1024*i);
	}
	normalize(arr);
	get_N_closest_from_prealigned_prenormalised_arrs_to_arr_given_prealigned_buf<10>(vecarr, num_rows, arr, closest_indices_arr);
}
