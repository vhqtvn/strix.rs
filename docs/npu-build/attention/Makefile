##===- Makefile (strix.rs fused-attention prototype) ----------------------===##
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
targetname=attention
# M=queries, K=keys/values(L), N=head_dim(D) — reuses makefile-common's M/K/N.
M?=64
K?=64
N?=64
kernels=attention
aieargs+=-M $M -K $K -N $N
target_suffix=${M}x${K}x${N}
use_placed?=0
use_iron?=0
use_linear_buf_alloc?=1
ifeq (${use_linear_buf_alloc}, 1)
buffer_aloc_flag=basic-sequential
else
buffer_aloc_flag=bank-aware
endif
aie_py_src=${targetname}.py
KERNEL_DEFINES=-DATT_M=${M} -DATT_L=${K} -DATT_D=${N}

include ${srcdir}/../matrix_multiplication/makefile-common

# makefile-common's kernels_dir (4 '../') is calibrated for the deeper
# matrix_multiplication/<subdir> path; attention/ is one level shallower, so use
# the correct relative path (3 '../' from basic/attention → mlir-aie root).
att_kernels_dir=${srcdir}/../../../aie_kernels/aie2p
build/attention.o: ${att_kernels_dir}/attention.cc
	mkdir -p ${@D}
	cd ${@D} && ${KERNEL_CC} ${KERNEL_CFLAGS} ${KERNEL_DEFINES} -c $< -o ${@F}
