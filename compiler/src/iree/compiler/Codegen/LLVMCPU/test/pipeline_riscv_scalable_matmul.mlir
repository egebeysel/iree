// Scalable RISC-V (RVV) data-tiling pipeline test: a data-tiled f32 matmul on
// encoded operands is materialized to mmt4d and lowered through the LLVMCPU
// configuration + lowering pipelines with scalable vectorization. The scalable
// mmt4d vectorizes WITHOUT masks (mirrors run_matmul_example.sh; the target
// triple/cpu-features live in the hal.executable.target attribute).
//
// Stage 1 materializes the device encodings into mmt4d; stage 2 runs the LLVMCPU
// configuration + lowering pipelines. Materialization must be func.func-nested,
// so it is a separate iree-opt invocation piped into the pipeline stage.
//
// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" \
// RUN:   --iree-llvmcpu-enable-scalable-vectorization=true %s \
// RUN: | iree-opt --split-input-file \
// RUN:   --iree-codegen-llvmcpu-configuration-pipeline \
// RUN:   --iree-codegen-llvmcpu-lowering-pipeline='include-llvm-lowering=false' \
// RUN:   --iree-llvmcpu-enable-scalable-vectorization=true \
// RUN:   --iree-experimental-vscale-value=4 \
// RUN: | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [384, 256, 512]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [384, 256, 512]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [384, 256, 512]>
#target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu = "", cpu_features = "+m,+a,+f,+d,+v,+zvfh", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 32 : i64, target_abi = "lp64d", target_triple = "riscv64-unknown-unknown-eabi-elf", ukernels = "none"}>
func.func @matmul_lowering_f32f32f32_riscv64() attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x512xf32, #encoding_lhs>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x256xf32, #encoding_rhs>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<384x256xf32, #encoding_result>>
  %lhs = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [384, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x512xf32, #encoding_lhs>> -> tensor<384x512xf32, #encoding_lhs>
  %rhs = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x256xf32, #encoding_rhs>> -> tensor<512x256xf32, #encoding_rhs>
  %init = tensor.empty() : tensor<384x256xf32, #encoding_result>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<384x256xf32, #encoding_result>) -> tensor<384x256xf32, #encoding_result>
  %res = linalg.matmul ins(%lhs, %rhs : tensor<384x512xf32, #encoding_lhs>, tensor<512x256xf32, #encoding_rhs>) outs(%fill : tensor<384x256xf32, #encoding_result>) -> tensor<384x256xf32, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %res, %2, offsets = [0, 0], sizes = [384, 256], strides = [1, 1] : tensor<384x256xf32, #encoding_result> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<384x256xf32, #encoding_result>>
  return
}

// CHECK-LABEL: func.func @matmul_lowering_f32f32f32_riscv64
// CHECK:         scf.for %{{.+}} = %c0 to %c512 step %c1 iter_args(%{{.+}} = %{{.+}}) -> (vector<7x[8]xf32>)
// The scalable mmt4d microkernel issues a single unmasked scalable RHS load and
// broadcasts 7 LHS scalars into 7 FMAs; no masking is needed (the load is a
// plain vector.load, not a vector.maskedload).
// CHECK:           %[[RHS:.+]] = vector.load {{.*}}, vector<[8]xf32>
// CHECK-COUNT-7:   vector.fma %{{.+}}, %[[RHS]], %{{.+}} : vector<[8]xf32>
// CHECK-NOT:       vector.maskedload
