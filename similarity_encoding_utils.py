import numpy as np
import ctypes


cosine_similarity_from_numpy_contiguous_array = None
write_to_file = None
load_from_file = None
get_10_closest_from_arrs_to_arr = None


def init(similarity_encoding_SO_file_path:str, encodings_cache_dir:str):
	global cosine_similarity_from_numpy_contiguous_array
	global write_to_file
	global load_from_file
	global get_10_closest_from_arrs_to_arr
	
	clib = ctypes.CDLL(similarity_encoding_SO_file_path)
	set_path_buf_dir = clib.set_path_buf_dir
	set_path_buf_dir.argtypes = [ctypes.c_char_p]
	set_path_buf_dir(encodings_cache_dir.encode())
	cosine_similarity_from_numpy_contiguous_array = clib.cosine_similarity_from_numpy_contiguous_array
	cosine_similarity_from_numpy_contiguous_array.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_ulong]
	cosine_similarity_from_numpy_contiguous_array.restype = ctypes.POINTER(ctypes.c_float)
	write_to_file = clib.write_to_file
	write_to_file.argtypes = [ctypes.c_ulong, ctypes.POINTER(ctypes.c_float), ctypes.c_uint]
	#write_to_file.restype  = ctypes.c_void
	load_from_file = clib.load_from_file
	load_from_file.argtypes = [ctypes.c_ulong, ctypes.POINTER(ctypes.c_float), ctypes.c_uint]
	load_from_file.restype  = ctypes.c_int
	get_10_closest_from_arrs_to_arr = clib.get_10_closest_from_arrs_to_arr
	get_10_closest_from_arrs_to_arr.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_ulong, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint)]


def encode_new_img_if_not_exist(cnn_encoder, filepath:str, fileid:int):
	c_arr = (ctypes.c_float * 1024)()
	success:bool = (load_from_file(fileid, c_arr, 1) == 0)
	errors:bool = False
	if not success:
		print("Encoding",filepath)
		encoding = cnn_encoder.encode_image(image_file=filepath)
		if encoding is None:
			errors = True
		elif fileid is not None:
			write_to_file(
				fileid,
				get_float_array_from_numpy_array(encoding),
				1
			)
	return errors


def get_float_array_from_numpy_array(array):
	float_array = np.ascontiguousarray(array, dtype=np.float32)
	float_array_ptr = float_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
	return float_array_ptr


def read_c_array(c_array_ptr, w:int, h:int):
	c_array = np.ctypeslib.as_array(c_array_ptr, shape=(w,h))
	print(c_array)
	return c_array.reshape(w, h)


def get_cosine_similarity(arr):
	n_elements:int = arr.shape[0]
	res_c_arr = cosine_similarity_from_numpy_contiguous_array(
		get_float_array_from_numpy_array(arr),
		n_elements
	)
	return read_c_array(res_c_arr, n_elements, n_elements)


def compare_similarity_scores_of_single_file(file_encodings:list, fileid:int):
	n_elements:int = len(file_encodings)
	file_contents = (ctypes.c_float * 1024)()
	success:bool = (load_from_file(fileid, file_contents, 1) == 0)
	closest_indices_arr = (ctypes.c_uint * 10)()
	get_10_closest_from_arrs_to_arr(
		get_float_array_from_numpy_array(np.array(file_encodings)),
		n_elements,
		file_contents,
		closest_indices_arr
	)
	print(np.ndarray(closest_indices_arr))
