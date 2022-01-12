#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

set(_IMPORT_PREFIX ${_IMPORT_PREFIX}/opencv/opencv4.1.0/arm64-v8a)

# Import target "libcpufeatures" for configuration "Release"
set_property(TARGET libcpufeatures APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libcpufeatures PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/3rdparty/libs/libcpufeatures.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS libcpufeatures )
list(APPEND _IMPORT_CHECK_FILES_FOR_libcpufeatures "${_IMPORT_PREFIX}/3rdparty/libs/libcpufeatures.a" )

# Import target "libjpeg-turbo" for configuration "Release"
set_property(TARGET libjpeg-turbo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libjpeg-turbo PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/3rdparty/libs/liblibjpeg-turbo.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS libjpeg-turbo )
list(APPEND _IMPORT_CHECK_FILES_FOR_libjpeg-turbo "${_IMPORT_PREFIX}/3rdparty/libs/liblibjpeg-turbo.a" )

# Import target "libtiff" for configuration "Release"
set_property(TARGET libtiff APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libtiff PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C;CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/3rdparty/libs/liblibtiff.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS libtiff )
list(APPEND _IMPORT_CHECK_FILES_FOR_libtiff "${_IMPORT_PREFIX}/3rdparty/libs/liblibtiff.a" )

# Import target "libwebp" for configuration "Release"
set_property(TARGET libwebp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libwebp PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/3rdparty/libs/liblibwebp.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS libwebp )
list(APPEND _IMPORT_CHECK_FILES_FOR_libwebp "${_IMPORT_PREFIX}/3rdparty/libs/liblibwebp.a" )

# Import target "libjasper" for configuration "Release"
set_property(TARGET libjasper APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libjasper PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/3rdparty/libs/liblibjasper.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS libjasper )
list(APPEND _IMPORT_CHECK_FILES_FOR_libjasper "${_IMPORT_PREFIX}/3rdparty/libs/liblibjasper.a" )

# Import target "libpng" for configuration "Release"
set_property(TARGET libpng APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libpng PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/3rdparty/libs/liblibpng.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS libpng )
list(APPEND _IMPORT_CHECK_FILES_FOR_libpng "${_IMPORT_PREFIX}/3rdparty/libs/liblibpng.a" )

# Import target "IlmImf" for configuration "Release"
set_property(TARGET IlmImf APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(IlmImf PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/3rdparty/libs/libIlmImf.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS IlmImf )
list(APPEND _IMPORT_CHECK_FILES_FOR_IlmImf "${_IMPORT_PREFIX}/3rdparty/libs/libIlmImf.a" )

# Import target "tbb" for configuration "Release"
set_property(TARGET tbb APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(tbb PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/3rdparty/libs/libtbb.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS tbb )
list(APPEND _IMPORT_CHECK_FILES_FOR_tbb "${_IMPORT_PREFIX}/3rdparty/libs/libtbb.a" )

# Import target "libprotobuf" for configuration "Release"
set_property(TARGET libprotobuf APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libprotobuf PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/3rdparty/libs/liblibprotobuf.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS libprotobuf )
list(APPEND _IMPORT_CHECK_FILES_FOR_libprotobuf "${_IMPORT_PREFIX}/3rdparty/libs/liblibprotobuf.a" )

# Import target "quirc" for configuration "Release"
set_property(TARGET quirc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(quirc PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/3rdparty/libs/libquirc.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS quirc )
list(APPEND _IMPORT_CHECK_FILES_FOR_quirc "${_IMPORT_PREFIX}/3rdparty/libs/libquirc.a" )

# Import target "tegra_hal" for configuration "Release"
set_property(TARGET tegra_hal APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(tegra_hal PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/3rdparty/libs/libtegra_hal.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS tegra_hal )
list(APPEND _IMPORT_CHECK_FILES_FOR_tegra_hal "${_IMPORT_PREFIX}/3rdparty/libs/libtegra_hal.a" )

# Import target "opencv_core" for configuration "Release"
set_property(TARGET opencv_core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_core PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/libs/libopencv_core.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_core )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_core "${_IMPORT_PREFIX}/libs/libopencv_core.a" )

# Import target "opencv_flann" for configuration "Release"
set_property(TARGET opencv_flann APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_flann PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/libs/libopencv_flann.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_flann )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_flann "${_IMPORT_PREFIX}/libs/libopencv_flann.a" )

# Import target "opencv_imgproc" for configuration "Release"
set_property(TARGET opencv_imgproc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_imgproc PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/libs/libopencv_imgproc.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_imgproc )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_imgproc "${_IMPORT_PREFIX}/libs/libopencv_imgproc.a" )

# Import target "opencv_ml" for configuration "Release"
set_property(TARGET opencv_ml APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_ml PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/libs/libopencv_ml.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_ml )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_ml "${_IMPORT_PREFIX}/libs/libopencv_ml.a" )

# Import target "opencv_photo" for configuration "Release"
set_property(TARGET opencv_photo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_photo PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/libs/libopencv_photo.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_photo )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_photo "${_IMPORT_PREFIX}/libs/libopencv_photo.a" )

# Import target "opencv_dnn" for configuration "Release"
set_property(TARGET opencv_dnn APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_dnn PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/libs/libopencv_dnn.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_dnn )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_dnn "${_IMPORT_PREFIX}/libs/libopencv_dnn.a" )

# Import target "opencv_imgcodecs" for configuration "Release"
set_property(TARGET opencv_imgcodecs APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_imgcodecs PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/libs/libopencv_imgcodecs.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_imgcodecs )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_imgcodecs "${_IMPORT_PREFIX}/libs/libopencv_imgcodecs.a" )

# Import target "opencv_videoio" for configuration "Release"
set_property(TARGET opencv_videoio APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_videoio PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/libs/libopencv_videoio.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_videoio )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_videoio "${_IMPORT_PREFIX}/libs/libopencv_videoio.a" )

# Import target "opencv_highgui" for configuration "Release"
set_property(TARGET opencv_highgui APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_highgui PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/libs/libopencv_highgui.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_highgui )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_highgui "${_IMPORT_PREFIX}/libs/libopencv_highgui.a" )

# Import target "opencv_features2d" for configuration "Release"
set_property(TARGET opencv_features2d APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_features2d PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/libs/libopencv_features2d.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_features2d )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_features2d "${_IMPORT_PREFIX}/libs/libopencv_features2d.a" )

# Import target "opencv_calib3d" for configuration "Release"
set_property(TARGET opencv_calib3d APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_calib3d PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/libs/libopencv_calib3d.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_calib3d )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_calib3d "${_IMPORT_PREFIX}/libs/libopencv_calib3d.a" )

# Import target "opencv_objdetect" for configuration "Release"
set_property(TARGET opencv_objdetect APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_objdetect PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/libs/libopencv_objdetect.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_objdetect )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_objdetect "${_IMPORT_PREFIX}/libs/libopencv_objdetect.a" )

# Import target "opencv_stitching" for configuration "Release"
set_property(TARGET opencv_stitching APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_stitching PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/libs/libopencv_stitching.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_stitching )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_stitching "${_IMPORT_PREFIX}/libs/libopencv_stitching.a" )

# Import target "opencv_video" for configuration "Release"
set_property(TARGET opencv_video APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_video PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/libs/libopencv_video.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_video )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_video "${_IMPORT_PREFIX}/libs/libopencv_video.a" )

# Import target "opencv_java" for configuration "Release"
set_property(TARGET opencv_java APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_java PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/libs/libopencv_java4.so"
  IMPORTED_SONAME_RELEASE "libopencv_java4.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_java )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_java "${_IMPORT_PREFIX}/libs/libopencv_java4.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
