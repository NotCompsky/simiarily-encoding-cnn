default: lib.so

lib.so:
	c++ lib.cpp -O3 -fPIC -shared -std=c++2a -o lib.so -march=native
