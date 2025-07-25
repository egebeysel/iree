################################################################################
# Autogenerated by build_tools/bazel_to_cmake/bazel_to_cmake.py from           #
# runtime/src/iree/base/internal/BUILD.bazel                                   #
#                                                                              #
# Use iree_cmake_extra_content from iree/build_defs.oss.bzl to add arbitrary   #
# CMake-only content.                                                          #
#                                                                              #
# To disable autogeneration for this file entirely, delete this header.        #
################################################################################

iree_add_all_subdirs()

iree_cc_library(
  NAME
    internal
  HDRS
    "atomics.h"
    "debugging.h"
    "inline_array.h"
    "math.h"
  SRCS
    "atomics_clang.h"
    "atomics_disabled.h"
    "atomics_gcc.h"
    "atomics_msvc.h"
  DEPS
    iree::base::core_headers
  PUBLIC
)

iree_cc_test(
  NAME
    atomics_test
  SRCS
    "atomics_test.cc"
  DEPS
    ::internal
    iree::base::core_headers
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    math_test
  SRCS
    "math_test.cc"
  DEPS
    ::internal
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    arena
  HDRS
    "arena.h"
  SRCS
    "arena.c"
  DEPS
    ::atomic_slist
    ::internal
    ::synchronization
    iree::base
  PUBLIC
)

iree_cc_library(
  NAME
    atomic_slist
  HDRS
    "atomic_slist.h"
  SRCS
    "atomic_slist.c"
  DEPS
    ::internal
    ::synchronization
    iree::base::core_headers
  PUBLIC
)

iree_cc_test(
  NAME
    atomic_slist_test
  SRCS
    "atomic_slist_test.cc"
  DEPS
    ::atomic_slist
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    cpu
  HDRS
    "cpu.h"
  SRCS
    "cpu.c"
  DEPS
    iree::base
    iree::base::core_headers
    iree::schemas::cpu_data
  PUBLIC
)

iree_cc_library(
  NAME
    dynamic_library
  HDRS
    "dynamic_library.h"
  SRCS
    "dynamic_library_posix.c"
    "dynamic_library_win32.c"
  DEPS
    ${CMAKE_DL_LIBS}
    ::internal
    ::path
    ::synchronization
    iree::base
  PUBLIC
)

iree_cc_library(
  NAME
    flags
  HDRS
    "flags.h"
  SRCS
    "flags.c"
  DEPS
    ::internal
    iree::base
    iree::io::file_handle
  PUBLIC
)

iree_cc_binary(
  NAME
    flags_demo
  SRCS
    "flags_demo.c"
  DEPS
    ::flags
    iree::base
  TESTONLY
)

iree_lit_test_suite(
  NAME
    flags_test
  SRCS
    "flags_test.txt"
  TOOLS
    ::flags_demo
    FileCheck
  LABELS
    "hostonly"
)

iree_cc_library(
  NAME
    fpu_state
  HDRS
    "fpu_state.h"
  SRCS
    "fpu_state.c"
  DEPS
    ::internal
    iree::base
    iree::base::core_headers
  PUBLIC
)

iree_cc_binary_benchmark(
  NAME
    fpu_state_benchmark
  SRCS
    "fpu_state_benchmark.cc"
  DEPS
    ::fpu_state
    iree::base
    iree::testing::benchmark
    iree::testing::benchmark_main
  TESTONLY
)

iree_cc_test(
  NAME
    fpu_state_test
  SRCS
    "fpu_state_test.cc"
  DEPS
    ::fpu_state
    iree::testing::gtest
    iree::testing::gtest_main
  LABELS
    "requires-dtz"
)

iree_cc_library(
  NAME
    memory
  HDRS
    "memory.h"
  SRCS
    "memory.c"
  DEPS
    ::internal
    iree::base
  PUBLIC
)

iree_cc_library(
  NAME
    path
  HDRS
    "path.h"
  SRCS
    "path.c"
  DEPS
    iree::base
    iree::base::core_headers
  PUBLIC
)

iree_cc_test(
  NAME
    path_test
  SRCS
    "path_test.cc"
  DEPS
    ::path
    iree::base::core_headers
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    prng
  HDRS
    "prng.h"
  DEPS
    ::internal
    iree::base::core_headers
  PUBLIC
)

iree_cc_test(
  NAME
    prng_test
  SRCS
    "prng_test.cc"
  DEPS
    ::prng
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    span
  HDRS
    "span.h"
  DEPS

  PUBLIC
)

iree_cc_library(
  NAME
    synchronization
  HDRS
    "call_once.h"
    "synchronization.h"
  SRCS
    "synchronization.c"
  DEPS
    ::internal
    iree::base
    iree::base::core_headers
  PUBLIC
)

iree_cc_binary_benchmark(
  NAME
    synchronization_benchmark
  SRCS
    "synchronization_benchmark.cc"
  DEPS
    ::synchronization
    benchmark
    iree::testing::benchmark_main
  TESTONLY
)

iree_cc_test(
  NAME
    synchronization_test
  SRCS
    "synchronization_test.cc"
  DEPS
    ::synchronization
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    time
  HDRS
    "time.h"
  SRCS
    "time.c"
  DEPS
    iree::base::core_headers
  PUBLIC
)

iree_cc_library(
  NAME
    wait_handle
  HDRS
    "wait_handle.h"
  SRCS
    "wait_handle.c"
    "wait_handle_emscripten.c"
    "wait_handle_epoll.c"
    "wait_handle_impl.h"
    "wait_handle_inproc.c"
    "wait_handle_kqueue.c"
    "wait_handle_null.c"
    "wait_handle_poll.c"
    "wait_handle_posix.c"
    "wait_handle_posix.h"
    "wait_handle_win32.c"
  DEPS
    ::synchronization
    iree::base
    iree::base::core_headers
  PUBLIC
)

iree_cc_test(
  NAME
    wait_handle_test
  SRCS
    "wait_handle_test.cc"
  DEPS
    ::wait_handle
    iree::testing::gtest
    iree::testing::gtest_main
)

if(NOT IREE_ENABLE_THREADING)
  return()
endif()

iree_cc_library(
  NAME
    event_pool
  HDRS
    "event_pool.h"
  SRCS
    "event_pool.c"
  DEPS
    ::internal
    ::synchronization
    ::wait_handle
    iree::base
    iree::base::core_headers
  PUBLIC
)

iree_cc_library(
  NAME
    threading
  HDRS
    "threading.h"
  SRCS
    "threading.c"
    "threading_darwin.c"
    "threading_impl.h"
    "threading_pthreads.c"
    "threading_win32.c"
  DEPS
    ${CMAKE_DL_LIBS}
    ::internal
    ::synchronization
    iree::base
    iree::base::core_headers
  PUBLIC
)

iree_cc_test(
  NAME
    threading_test
  SRCS
    "threading_impl.h"
    "threading_test.cc"
  DEPS
    ::internal
    ::synchronization
    ::threading
    iree::base
    iree::testing::gtest
    iree::testing::gtest_main
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###

if(EMSCRIPTEN)
  iree_link_js_library(
    TARGET
      ::wait_handle
    SRCS
      "wait_handle_emscripten.js"
  )
endif()
