##===- Makefile (strix.rs fused-attention prototype) ----------------------===##
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
targetname=attention
# M=queries, K=keys/values(L), N=head_dim(D) — reuses makefile-common's M/K/N.
# M = Mtile (rows per kernel call, == ATT_M). MQ = total queries (default = Mtile,
# i.e. no query tiling). K = L (keys/values). N = D (head dim). LB = KV block size.
M?=64
MQ?=${M}
K?=64
N?=64
LB?=32
KVDEPTH?=2
kernels=attention
aieargs+=-M $M --mq ${MQ} -K $K -N $N --lb ${LB} --kvdepth ${KVDEPTH}
target_suffix=${MQ}x${K}x${N}_m${M}
use_placed?=0
use_iron?=0
use_linear_buf_alloc?=1
ifeq (${use_linear_buf_alloc}, 1)
buffer_aloc_flag=basic-sequential
else
buffer_aloc_flag=bank-aware
endif
aie_py_src=${targetname}.py
KERNEL_DEFINES=-DATT_M=${M} -DATT_L=${K} -DATT_D=${N} -DATT_LB=${LB}

include ${srcdir}/../matrix_multiplication/makefile-common

# makefile-common's kernels_dir (4 '../') is calibrated for the deeper
# matrix_multiplication/<subdir> path; attention/ is one level shallower, so use
# the correct relative path (3 '../' from basic/attention → mlir-aie root).
att_kernels_dir=${srcdir}/../../../aie_kernels/aie2p
build/attention.o: ${att_kernels_dir}/attention.cc
	mkdir -p ${@D}
	cd ${@D} && ${KERNEL_CC} ${KERNEL_CFLAGS} ${KERNEL_DEFINES} -c $< -o ${@F}
