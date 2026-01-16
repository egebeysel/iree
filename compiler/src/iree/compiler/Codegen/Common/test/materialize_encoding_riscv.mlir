// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" --split-input-file %s | FileCheck %s --check-prefixes=CHECK,NO-RVV
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" --iree-llvmcpu-enable-scalable-vectorization=true --split-input-file %s | FileCheck %s --check-prefixes=CHECK,WITH-RVV

// Tests for RISC-V targets with and without scalable vectorization.
// RISC-V64 with the V extension supports scalable tiles where the N dimension
// scales with vscale (VLEN/64). The base tile sizes assume VLEN=64 bits.

//===----------------------------------------------------------------------===//
// RISC-V32 without V extension - no data tiling
//===----------------------------------------------------------------------===//

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_f32f32f32_riscv32_no_v_ext(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %acc: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv32-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = tensor.dim %acc, %c0 : tensor<?x?xf32>
  %N = tensor.dim %acc, %c1 : tensor<?x?xf32>
  %0 = iree_encoding.set_encoding %lhs : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_lhs>
  %1 = iree_encoding.set_encoding %rhs : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_rhs>
  %2 = iree_encoding.set_encoding %acc : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_result>
  %3 = linalg.matmul
      ins(%0, %1 : tensor<?x?xf32, #encoding_lhs>,
                   tensor<?x?xf32, #encoding_rhs>)
      outs(%2 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  %4 = iree_encoding.unset_encoding %3 : tensor<?x?xf32, #encoding_result> -> tensor<?x?xf32>{%M, %N}
  return %4 : tensor<?x?xf32>
}
// RISC-V32 without V extension does not implement data-tiling.
// CHECK-LABEL: func @matmul_lowering_f32f32f32_riscv32_no_v_ext
//       CHECK:   %[[RES:.+]] = linalg.matmul
//       CHECK:   return %[[RES]]

// -----

//===----------------------------------------------------------------------===//
// RISC-V32 with ukernels - uses mmt4d with default tiles
//===----------------------------------------------------------------------===//

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i8i32_riscv32_ukernel(
    %lhs: tensor<?x?xi8, #encoding_lhs>,
    %rhs: tensor<?x?xi8, #encoding_rhs>,
    %result: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv32-xyz-xyz", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %out = linalg.matmul
      ins(%lhs, %rhs : tensor<?x?xi8, #encoding_lhs>,
                       tensor<?x?xi8, #encoding_rhs>)
      outs(%result : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  return %out : tensor<?x?xi32, #encoding_result>
}
// CHECK-LABEL: func @matmul_lowering_i8i8i32_riscv32_ukernel(
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x8x4xi8>
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x8x4xi8>
// CHECK-SAME:    %[[ACC:[a-zA-Z0-9]+]]: tensor<?x?x8x8xi32>
// CHECK:         %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:      ins(%[[LHS]], %[[RHS]]
// CHECK-SAME:      outs(%[[ACC]]
// CHECK:         return %[[MMT4D]]

// -----

//===----------------------------------------------------------------------===//
// RISC-V64 with V extension - set_encoding for LHS (f32)
//===----------------------------------------------------------------------===//
// The LHS operand corresponds to the M dimension, which is NOT scalable.
// Therefore, NO-RVV and WITH-RVV produce the same result for LHS.

#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>], iteration_sizes = [?, ?, ?]>
func.func @matmul_set_encoding_LHS_f32_riscv64(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32, #encoding> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", cpu_features="+v", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %arg0 : tensor<8x16xf32> -> tensor<8x16xf32, #encoding>
  return %0 : tensor<8x16xf32, #encoding>
}

/// NOTE: No scalable tiles for LHS (M dimension), hence no difference between NO-RVV and WITH-RVV.
/// The inner tile for M is 7 (from enumerateMatmulTileRiscv64), K is 1.

// CHECK-LABEL: func.func @matmul_set_encoding_LHS_f32_riscv64
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<8x16xf32>
// CHECK:         %[[PACK:.+]] = linalg.pack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [7, 1]
// CHECK:         return %[[PACK]]

// -----

//===----------------------------------------------------------------------===//
// RISC-V64 with V extension - set_encoding for RHS (f32)
//===----------------------------------------------------------------------===//
// The RHS operand corresponds to the N dimension, which IS scalable.
// Therefore, NO-RVV and WITH-RVV differ in how the N dimension is tiled.

#encoding = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>], iteration_sizes = [?, ?, ?]>
func.func @matmul_set_encoding_RHS_f32_riscv64(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32, #encoding> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", cpu_features="+v", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %arg0 : tensor<8x16xf32> -> tensor<8x16xf32, #encoding>
  return %0 : tensor<8x16xf32, #encoding>
}

/// For RHS, the inner tile corresponding to the "N" dimension is scalable.
/// Base N tile for f32 with VLEN=64 is 2 (64 bits / 32 bits = 2 elements).
/// With scalable vectorization, the N tile becomes 2 * vscale.

// WITH-RVV: #[[$MAP:.+]] = affine_map<()[s0] -> (16 ceildiv s0)>

// CHECK-LABEL: func.func @matmul_set_encoding_RHS_f32_riscv64
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<8x16xf32>
// WITH-RVV-DAG:  %[[PAD:.+]] = arith.constant 0.000000e+00 : f32

/// RVV: the number of outer tiles corresponding to the inner scalable tile
// WITH-RVV-DAG:  %[[C2:.*]] = arith.constant 2 : index
// WITH-RVV-DAG:  %[[VSCALE:.*]] = vector.vscale
// WITH-RVV:      %[[C2_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C2]] : index
// WITH-RVV:      %[[OUTER_DIM:.*]] = affine.apply #[[$MAP]]()[%[[C2_VSCALE]]]

/// Init the output tensor
// NO-RVV-DAG:     %[[INIT:.+]] = tensor.empty() : tensor<8x8x2x1xf32>
// WITH-RVV-DAG:   %[[INIT:.*]] = tensor.empty(%[[OUTER_DIM]], %[[C2_VSCALE]]) : tensor<?x8x?x1xf32>

/// The newly materialised Pack Op (RVV includes padding for dynamic tile)
// CHECK:         %[[PACK:.+]] = linalg.pack %[[SRC]]
// WITH-RVV-SAME:    padding_value(%[[PAD]] : f32)
// NO-RVV-NOT:        padding_value

// CHECK-SAME:      outer_dims_perm = [1, 0]
// CHECK-SAME:      inner_dims_pos = [1, 0]

// NO-RVV-SAME:      inner_tiles = [2, 1]
// NO-RVV-SAME:      into %[[INIT]] : tensor<8x16xf32> -> tensor<8x8x2x1xf32>

// WITH-RVV-SAME:    inner_tiles = [%[[C2_VSCALE]], 1]
// WITH-RVV-SAME:    into %[[INIT]] : tensor<8x16xf32> -> tensor<?x8x?x1xf32>

// CHECK:         return %[[PACK]]

// -----

//===----------------------------------------------------------------------===//
// RISC-V64 with V extension - unset_encoding for RESULT (f32)
//===----------------------------------------------------------------------===//
// The RESULT operand has both M and N dimensions. M is fixed, N is scalable.

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [80, 320, ?]>
func.func @matmul_unset_encoding_RESULT_f32_riscv64(%arg0: tensor<80x320xf32, #encoding>) -> tensor<80x320xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", cpu_features="+v", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.unset_encoding %arg0 : tensor<80x320xf32, #encoding> -> tensor<80x320xf32>
  return %0 : tensor<80x320xf32>
}

/// For RESULT, the inner tile for N is scalable with base size 2.
/// M tile is 7 (fixed), N tile is 2 * vscale.

// CHECK-LABEL: func.func @matmul_unset_encoding_RESULT_f32_riscv64
// NO-RVV-SAME:    %[[INPUT:[a-zA-Z0-9]+]]: tensor<12x160x7x2xf32>
// WITH-RVV-SAME:  %[[INPUT:[a-zA-Z0-9]+]]: tensor<12x?x7x?xf32>

// CHECK-DAG:     %[[EMPTY:.+]] = tensor.empty()

/// RVV: compute the scalable tile size
// WITH-RVV-DAG:  %[[C2:.+]] = arith.constant 2 : index
// WITH-RVV:      %[[VSCALE:.+]] = vector.vscale
// WITH-RVV:      %[[C2_VSCALE:.+]] = arith.muli %[[VSCALE]], %[[C2]] : index

/// The newly materialised UnPack Op
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %[[INPUT]]
// NO-RVV-SAME:       outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [7, 2] into %[[EMPTY]]
// WITH-RVV-SAME:     outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [7, %[[C2_VSCALE]]] into %[[EMPTY]]

// CHECK:         return %[[UNPACK]]

// -----

//===----------------------------------------------------------------------===//
// RISC-V64 with V extension - full matmul lowering (f32)
//===----------------------------------------------------------------------===//

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_f32f32f32_riscv64(
    %lhs: tensor<?x?xf32, #encoding_lhs>,
    %rhs: tensor<?x?xf32, #encoding_rhs>,
    %result: tensor<?x?xf32, #encoding_result>
) -> tensor<?x?xf32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", cpu_features="+v", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %matmul = linalg.matmul
    ins(%lhs, %rhs : tensor<?x?xf32, #encoding_lhs>, tensor<?x?xf32, #encoding_rhs>)
    outs(%result : tensor<?x?xf32, #encoding_result>)
    -> tensor<?x?xf32, #encoding_result>
  return %matmul : tensor<?x?xf32, #encoding_result>
}

/// For f32 on RISC-V64 with V extension:
/// - M tile: 7 (fixed)
/// - N tile: 2 (scalable with vscale when RVV enabled)
/// - K tile: 1 (fixed)
/// LHS: [M, K] -> packed as [?, ?, 7, 1]
/// RHS: [K, N] -> packed as [?, ?, N_tile, 1] with outer_dims_perm = [1, 0]
/// RESULT: [M, N] -> packed as [?, ?, 7, N_tile]

// CHECK-LABEL: func @matmul_lowering_f32f32f32_riscv64(
// NO-RVV-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x7x1xf32>
// NO-RVV-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x2x1xf32>
// NO-RVV-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x7x2xf32>
// WITH-RVV-SAME: %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x7x1xf32>
// WITH-RVV-SAME: %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x?x1xf32>
// WITH-RVV-SAME: %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x7x?xf32>
// CHECK:         %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
// CHECK:         return %[[MMT4D]]

// -----

//===----------------------------------------------------------------------===//
// RISC-V64 with V extension and zvfh - matmul lowering (f16)
//===----------------------------------------------------------------------===//

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_f16f16f16_riscv64(
    %lhs: tensor<?x?xf16, #encoding_lhs>,
    %rhs: tensor<?x?xf16, #encoding_rhs>,
    %result: tensor<?x?xf16, #encoding_result>
) -> tensor<?x?xf16, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", cpu_features="+v,+zvfh", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %matmul = linalg.matmul
    ins(%lhs, %rhs : tensor<?x?xf16, #encoding_lhs>, tensor<?x?xf16, #encoding_rhs>)
    outs(%result : tensor<?x?xf16, #encoding_result>)
    -> tensor<?x?xf16, #encoding_result>
  return %matmul : tensor<?x?xf16, #encoding_result>
}

/// For f16 on RISC-V64 with V+zvfh extension:
/// - M tile: 7 (fixed)
/// - N tile: 4 (scalable with vscale when RVV enabled, base = 64/16 = 4)
/// - K tile: 1 (fixed)

// CHECK-LABEL: func @matmul_lowering_f16f16f16_riscv64(
// NO-RVV-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x7x1xf16>
// NO-RVV-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x4x1xf16>
// NO-RVV-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x7x4xf16>
// WITH-RVV-SAME: %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x7x1xf16>
// WITH-RVV-SAME: %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x?x1xf16>
// WITH-RVV-SAME: %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x7x?xf16>
// CHECK:         %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
// CHECK:         return %[[MMT4D]]

// -----

//===----------------------------------------------------------------------===//
// RISC-V64 without V extension - no data tiling (fallback)
//===----------------------------------------------------------------------===//

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_f32f32f32_riscv64_no_v_ext(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %acc: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = tensor.dim %acc, %c0 : tensor<?x?xf32>
  %N = tensor.dim %acc, %c1 : tensor<?x?xf32>
  %0 = iree_encoding.set_encoding %lhs : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_lhs>
  %1 = iree_encoding.set_encoding %rhs : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_rhs>
  %2 = iree_encoding.set_encoding %acc : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_result>
  %3 = linalg.matmul
      ins(%0, %1 : tensor<?x?xf32, #encoding_lhs>,
                   tensor<?x?xf32, #encoding_rhs>)
      outs(%2 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  %4 = iree_encoding.unset_encoding %3 : tensor<?x?xf32, #encoding_result> -> tensor<?x?xf32>{%M, %N}
  return %4 : tensor<?x?xf32>
}

// RISC-V64 without V extension does not implement data-tiling.
// CHECK-LABEL: func @matmul_lowering_f32f32f32_riscv64_no_v_ext
//       CHECK:   %[[RES:.+]] = linalg.matmul
//       CHECK:   return %[[RES]]
