# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/ridhamap/monostructure_alpha_signal/hf-microstructure/build/_deps/nanobench-src")
  file(MAKE_DIRECTORY "/Users/ridhamap/monostructure_alpha_signal/hf-microstructure/build/_deps/nanobench-src")
endif()
file(MAKE_DIRECTORY
  "/Users/ridhamap/monostructure_alpha_signal/hf-microstructure/build/_deps/nanobench-build"
  "/Users/ridhamap/monostructure_alpha_signal/hf-microstructure/build/_deps/nanobench-subbuild/nanobench-populate-prefix"
  "/Users/ridhamap/monostructure_alpha_signal/hf-microstructure/build/_deps/nanobench-subbuild/nanobench-populate-prefix/tmp"
  "/Users/ridhamap/monostructure_alpha_signal/hf-microstructure/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp"
  "/Users/ridhamap/monostructure_alpha_signal/hf-microstructure/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src"
  "/Users/ridhamap/monostructure_alpha_signal/hf-microstructure/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/ridhamap/monostructure_alpha_signal/hf-microstructure/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/ridhamap/monostructure_alpha_signal/hf-microstructure/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
