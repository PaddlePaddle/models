#ifndef OPENCV_CVCONFIG_H_INCLUDED
#define OPENCV_CVCONFIG_H_INCLUDED

/* OpenCV compiled as static or dynamic libs */
/* #undef BUILD_SHARED_LIBS */

/* OpenCV intrinsics optimized code */
#define CV_ENABLE_INTRINSICS

/* OpenCV additional optimized code */
/* #undef CV_DISABLE_OPTIMIZATION */

/* Compile for 'real' NVIDIA GPU architectures */
#define CUDA_ARCH_BIN ""

/* Create PTX or BIN for 1.0 compute capability */
/* #undef CUDA_ARCH_BIN_OR_PTX_10 */

/* NVIDIA GPU features are used */
#define CUDA_ARCH_FEATURES ""

/* Compile for 'virtual' NVIDIA PTX architectures */
#define CUDA_ARCH_PTX ""

/* AMD's Basic Linear Algebra Subprograms Library*/
/* #undef HAVE_CLAMDBLAS */

/* AMD's OpenCL Fast Fourier Transform Library*/
/* #undef HAVE_CLAMDFFT */

/* Clp support */
/* #undef HAVE_CLP */

/* Cocoa API */
/* #undef HAVE_COCOA */

/* NVIDIA CUDA Basic Linear Algebra Subprograms (BLAS) API*/
/* #undef HAVE_CUBLAS */

/* NVIDIA CUDA Runtime API*/
/* #undef HAVE_CUDA */

/* NVIDIA CUDA Fast Fourier Transform (FFT) API*/
/* #undef HAVE_CUFFT */

/* DirectX */
/* #undef HAVE_DIRECTX */
/* #undef HAVE_DIRECTX_NV12 */
/* #undef HAVE_D3D11 */
/* #undef HAVE_D3D10 */
/* #undef HAVE_D3D9 */

/* Eigen Matrix & Linear Algebra Library */
/* #undef HAVE_EIGEN */

/* Geospatial Data Abstraction Library */
/* #undef HAVE_GDAL */

/* GTK+ 2.0 Thread support */
/* #undef HAVE_GTHREAD */

/* GTK+ 2.x toolkit */
/* #undef HAVE_GTK */

/* Halide support */
/* #undef HAVE_HALIDE */

/* Vulkan support */
/* #undef HAVE_VULKAN */

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Intel Integrated Performance Primitives */
#define HAVE_IPP
#define HAVE_IPP_ICV
#define HAVE_IPP_IW
#define HAVE_IPP_IW_LL

/* JPEG-2000 codec */
#define HAVE_JASPER

/* IJG JPEG codec */
#define HAVE_JPEG

/* libpng/png.h needs to be included */
/* #undef HAVE_LIBPNG_PNG_H */

/* GDCM DICOM codec */
/* #undef HAVE_GDCM */

/* NVIDIA Video Decoding API*/
/* #undef HAVE_NVCUVID */

/* NVIDIA Video Encoding API*/
/* #undef HAVE_NVCUVENC */

/* OpenCL Support */
/* #undef HAVE_OPENCL */
/* #undef HAVE_OPENCL_STATIC */
/* #undef HAVE_OPENCL_SVM */

/* NVIDIA OpenCL D3D Extensions support */
/* #undef HAVE_OPENCL_D3D11_NV */

/* OpenEXR codec */
#define HAVE_OPENEXR

/* OpenGL support*/
/* #undef HAVE_OPENGL */

/* PNG codec */
#define HAVE_PNG

/* Posix threads (pthreads) */
#define HAVE_PTHREAD

/* parallel_for with pthreads */
#define HAVE_PTHREADS_PF

/* Qt support */
/* #undef HAVE_QT */

/* Qt OpenGL support */
/* #undef HAVE_QT_OPENGL */

/* Intel Threading Building Blocks */
#define HAVE_TBB

/* Ste||ar Group High Performance ParallelX */
/* #undef HAVE_HPX */

/* TIFF codec */
#define HAVE_TIFF

/* Win32 UI */
/* #undef HAVE_WIN32UI */

/* Define if your processor stores words with the most significant byte
   first (like Motorola and SPARC, unlike Intel and VAX). */
/* #undef WORDS_BIGENDIAN */

/* VA library (libva) */
/* #undef HAVE_VA */

/* Intel VA-API/OpenCL */
/* #undef HAVE_VA_INTEL */

/* Lapack */
/* #undef HAVE_LAPACK */

/* Library was compiled with functions instrumentation */
/* #undef ENABLE_INSTRUMENTATION */

/* OpenVX */
/* #undef HAVE_OPENVX */

/* OpenCV trace utilities */
#define OPENCV_TRACE

/* Library QR-code decoding */
#define HAVE_QUIRC

#endif // OPENCV_CVCONFIG_H_INCLUDED
