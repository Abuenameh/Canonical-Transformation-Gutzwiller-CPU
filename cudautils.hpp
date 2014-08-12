/*
 * cudautil.hpp
 *
 *  Created on: Aug 11, 2014
 *      Author: Abuenameh
 */

#ifndef CUDAUTIL_HPP_
#define CUDAUTIL_HPP_

#include <cuda.h>
#include <cuda_runtime.h>

inline int iAlignUp(int a, int b)
{
	return (a % b != 0) ?  (a - a % b + b) : a;
}

template<class T>
inline int memAlloc(T** ptr, size_t length) {
	void* ptr_;
	cudaMalloc(&ptr_, length * sizeof(T));
	*ptr = (T*)ptr_;
	if(ptr_ != NULL)
		return 0;
	else
		return -1;
}

template<class T>
inline int memAllocHost(T** ptr, T** ptr_dev, size_t length) {
	void* ptr_;
	cudaHostAlloc(&ptr_, length * sizeof(T), cudaHostAllocMapped);
	if(ptr_dev)
		cudaHostGetDevicePointer((void**)ptr_dev, ptr_, 0);
	*ptr = (T*)ptr_;

	memset(*ptr, 0, length);

	if(ptr_ != NULL)
		return 0;
	else
		return -1;
}

template<class T>
inline int memAllocPitch(T** ptr, size_t lx, size_t ly, size_t* pitch) {
	int bl = lx * sizeof(T);
	void* ptr_;
	bl = iAlignUp(bl, 32);
	if(pitch)
		*pitch = bl / sizeof(T);
	int length = bl * ly;
	cudaMalloc(&ptr_, length);
	*ptr = (T*)ptr_;

	cudaMemset(*ptr, 0, length);

	if(ptr_ != NULL)
		return 0;
	else
		return -1;
}

inline void memFree(void* ptr) {
	cudaFree(ptr);
}

inline void memFreeHost(void* ptr) {
	cudaFreeHost(ptr);
}

inline void memCopy(void* p1, const void* p2, size_t length, int type = 0) {
	cudaMemcpy(p1, p2, length, (cudaMemcpyKind)type);
}

inline void memCopyAsync(void* p1, const void* p2, size_t length, int type = 0, const cudaStream_t stream = NULL) {
	cudaMemcpyAsync(p1, p2, length, (cudaMemcpyKind)type, stream);
}

inline void memSet(void* p1, int _val, size_t length) {
	memset(p1, _val, length);
}



#endif /* CUDAUTIL_HPP_ */
