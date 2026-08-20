// Materialization of data-tiled f32 matmul encodings to iree_codegen.inner_tiled
// on RISC-V V. The intrinsic's N tile is `vlen / 8` (VLMAX for f32m4), so the
// same MMAIntrinsic enum value covers every VLEN: the concrete vector length is
// carried by the `vlen` parameter on the layout attribute, resolved from the
// target's `+zvl*b` features at selection time.
//
// Scalable vectorization is deliberately out of scope here — under
// `--iree-llvmcpu-enable-scalable-vectorization=true` the RVV intrinsic is not a
// candidate at all and `getScalableTileFlags` owns the layout instead, so those
// cases belong in materialize_encoding_riscv.mlir.

// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" --split-input-file %s | FileCheck %s

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_f32_inner_tiled_zvl128b(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %m: index, %n: index, %k: index) -> tensor<?x?xf32> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple = "riscv64-unknown-unknown-eabi-elf", cpu_features = "+64bit,+m,+a,+f,+d,+c,+v,+zvl128b", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #lhs>
  %1 = iree_encoding.set_encoding %arg1 encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf32, #acc>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xf32, #lhs>, tensor<?x?xf32, #rhs>)
      outs(%3 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %5 = iree_encoding.unset_encoding %4 encoding_dims{%m, %n, %k} : tensor<?x?xf32, #acc> -> tensor<?x?xf32>{%d0, %d1}
  return %5 : tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul_f32_inner_tiled_zvl128b(
//       CHECK:   %[[PACK_LHS:.+]] = linalg.pack {{.*}}inner_tiles = [6, 1]
//  CHECK-SAME:       -> tensor<?x?x6x1xf32>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[PACK_LHS]]
//  CHECK-SAME:       into tensor<?x?x6x1x1xf32>
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack {{.*}}inner_tiles = [16, 1]
//  CHECK-SAME:       -> tensor<?x?x16x1xf32>
//       CHECK:   %[[INNER:.+]] = iree_codegen.inner_tiled ins(%[[EXPANDED]], %[[PACK_RHS]])
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VFMACC_1xVLsx1_F32_F32, intrinsics_m = 6, vlen = 128>
//  CHECK-SAME:       tensor<?x?x6x1x1xf32>, tensor<?x?x16x1xf32> into tensor<?x?x6x1x16xf32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[INNER]]
//       CHECK:   linalg.unpack %[[COLLAPSED]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_f32_inner_tiled_zvl256b(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %m: index, %n: index, %k: index) -> tensor<?x?xf32> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple = "riscv64-unknown-unknown-eabi-elf", cpu_features = "+64bit,+m,+a,+f,+d,+c,+v,+zvl256b", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #lhs>
  %1 = iree_encoding.set_encoding %arg1 encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf32, #acc>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xf32, #lhs>, tensor<?x?xf32, #rhs>)
      outs(%3 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %5 = iree_encoding.unset_encoding %4 encoding_dims{%m, %n, %k} : tensor<?x?xf32, #acc> -> tensor<?x?xf32>{%d0, %d1}
  return %5 : tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul_f32_inner_tiled_zvl256b(
//       CHECK:   %[[PACK_LHS:.+]] = linalg.pack {{.*}}inner_tiles = [6, 1]
//  CHECK-SAME:       -> tensor<?x?x6x1xf32>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[PACK_LHS]]
//  CHECK-SAME:       into tensor<?x?x6x1x1xf32>
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack {{.*}}inner_tiles = [32, 1]
//  CHECK-SAME:       -> tensor<?x?x32x1xf32>
//       CHECK:   %[[INNER:.+]] = iree_codegen.inner_tiled ins(%[[EXPANDED]], %[[PACK_RHS]])
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VFMACC_1xVLsx1_F32_F32, intrinsics_m = 6, vlen = 256>
//  CHECK-SAME:       tensor<?x?x6x1x1xf32>, tensor<?x?x32x1xf32> into tensor<?x?x6x1x32xf32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[INNER]]
//       CHECK:   linalg.unpack %[[COLLAPSED]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_f32_inner_tiled_zvl512b(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %m: index, %n: index, %k: index) -> tensor<?x?xf32> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple = "riscv64-unknown-unknown-eabi-elf", cpu_features = "+64bit,+m,+a,+f,+d,+c,+v,+zvl512b", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #lhs>
  %1 = iree_encoding.set_encoding %arg1 encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf32, #acc>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xf32, #lhs>, tensor<?x?xf32, #rhs>)
      outs(%3 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %5 = iree_encoding.unset_encoding %4 encoding_dims{%m, %n, %k} : tensor<?x?xf32, #acc> -> tensor<?x?xf32>{%d0, %d1}
  return %5 : tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul_f32_inner_tiled_zvl512b(
//       CHECK:   %[[PACK_LHS:.+]] = linalg.pack {{.*}}inner_tiles = [6, 1]
//  CHECK-SAME:       -> tensor<?x?x6x1xf32>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[PACK_LHS]]
//  CHECK-SAME:       into tensor<?x?x6x1x1xf32>
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack {{.*}}inner_tiles = [64, 1]
//  CHECK-SAME:       -> tensor<?x?x64x1xf32>
//       CHECK:   %[[INNER:.+]] = iree_codegen.inner_tiled ins(%[[EXPANDED]], %[[PACK_RHS]])
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VFMACC_1xVLsx1_F32_F32, intrinsics_m = 6, vlen = 512>
//  CHECK-SAME:       tensor<?x?x6x1x1xf32>, tensor<?x?x64x1xf32> into tensor<?x?x6x1x64xf32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[INNER]]
//       CHECK:   linalg.unpack %[[COLLAPSED]]

// -----

// No `+zvl*b` at all: the V extension's architectural minimum VLEN is 128, so
// selection falls back to that rather than declining the intrinsic.

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_f32_inner_tiled_v_only(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %m: index, %n: index, %k: index) -> tensor<?x?xf32> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple = "riscv64-unknown-unknown-eabi-elf", cpu_features = "+64bit,+m,+a,+f,+d,+c,+v", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #lhs>
  %1 = iree_encoding.set_encoding %arg1 encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf32, #acc>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xf32, #lhs>, tensor<?x?xf32, #rhs>)
      outs(%3 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %5 = iree_encoding.unset_encoding %4 encoding_dims{%m, %n, %k} : tensor<?x?xf32, #acc> -> tensor<?x?xf32>{%d0, %d1}
  return %5 : tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul_f32_inner_tiled_v_only(
//       CHECK:   %[[PACK_LHS:.+]] = linalg.pack {{.*}}inner_tiles = [6, 1]
//  CHECK-SAME:       -> tensor<?x?x6x1xf32>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[PACK_LHS]]
//  CHECK-SAME:       into tensor<?x?x6x1x1xf32>
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack {{.*}}inner_tiles = [16, 1]
//  CHECK-SAME:       -> tensor<?x?x16x1xf32>
//       CHECK:   %[[INNER:.+]] = iree_codegen.inner_tiled ins(%[[EXPANDED]], %[[PACK_RHS]])
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VFMACC_1xVLsx1_F32_F32, intrinsics_m = 6, vlen = 128>
//  CHECK-SAME:       tensor<?x?x6x1x1xf32>, tensor<?x?x16x1xf32> into tensor<?x?x6x1x16xf32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[INNER]]
//       CHECK:   linalg.unpack %[[COLLAPSED]]
