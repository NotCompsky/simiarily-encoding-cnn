import numpy as np
import ctypes


encodings_cache_dir:str = "/media/vangelic/DATA/CNN_encodings_cache_dir"
libby = ctypes.CDLL("lib.so")
cosine_similarity_from_numpy_contiguous_array = libby.cosine_similarity_from_numpy_contiguous_array
cosine_similarity_from_numpy_contiguous_array.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
cosine_similarity_from_numpy_contiguous_array.restype = ctypes.POINTER(ctypes.c_float)

def get_float_array_from_numpy_array(array):
	float_array = np.ascontiguousarray(array, dtype=np.float32)
	float_array_ptr = float_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
	return float_array_ptr

def read_c_array(c_array_ptr, w:int, h:int):
	c_array = np.ctypeslib.as_array(c_array_ptr, shape=(w,h))
	print(c_array)
	return c_array.reshape(w, h)


if __name__ == "__main__":
	import os
	import pickle
	
	file_encodings:list = []
	n_elements:int = 0
	for fname in os.listdir(encodings_cache_dir):
		file_encodings.append(pickle.load(open(f"{encodings_cache_dir}/{fname}","rb")))
		n_elements += 1
		if n_elements == 100:
			break
	arr = np.array(file_encodings)
	print(arr)
	res_c_arr = cosine_similarity_from_numpy_contiguous_array(
		get_float_array_from_numpy_array(arr),
		n_elements
	)
	arr = read_c_array(res_c_arr, n_elements, n_elements)
	print(arr)
