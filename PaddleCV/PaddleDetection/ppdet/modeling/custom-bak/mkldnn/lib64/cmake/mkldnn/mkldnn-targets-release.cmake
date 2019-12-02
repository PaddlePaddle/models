#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "MKLDNN::mkldnn" for configuration "Release"
set_property(TARGET MKLDNN::mkldnn APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MKLDNN::mkldnn PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib64/libmkldnn.so.0.18.0.0"
  IMPORTED_SONAME_RELEASE "libmkldnn.so.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS MKLDNN::mkldnn )
list(APPEND _IMPORT_CHECK_FILES_FOR_MKLDNN::mkldnn "${_IMPORT_PREFIX}/lib64/libmkldnn.so.0.18.0.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
