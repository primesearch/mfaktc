# where is the CUDA Toolkit installed?
CUDA_DIR = /usr/local/cuda
CUDA_INCLUDE = -I$(CUDA_DIR)/include/
CUDA_LIB = -L$(CUDA_DIR)/lib64/

# Compiler settings for .c files (CPU)
CC = gcc
CFLAGS = -Wall -Wextra -O2 $(CUDA_INCLUDE) -malign-double
CFLAGS_EXTRA_SIEVE = -funroll-all-loops

# Compiler settings for .cu files (CPU/GPU)
NVCC = nvcc
NVCCFLAGS = $(CUDA_INCLUDE) --ptxas-options=-v

# generate code for various compute capabilities
# don't try "arch=compute_1x,code=sm_1x". WON'T WORK! Stay with mfaktc 0.21 for CC 1.1, 1.2 and 1.3 GPUs.
# NVCCFLAGS += --generate-code arch=compute_20,code=sm_20 # CC 2.x GPUs will use this code, one code fits all!
# NVCCFLAGS += --generate-code arch=compute_30,code=sm_30 # all CC 3.x GPUs _COULD_ use this code 
# NVCCFLAGS += --generate-code arch=compute_35,code=sm_35 # but CC 3.5 (3.2?) _CAN_ use funnel shift which is useful for mfaktc
# NVCCFLAGS += --generate-code arch=compute_50,code=sm_50 # CC 5.x GPUs (Maxwell) will use this code
NVCCFLAGS += --generate-code arch=compute_60,code=sm_60 # CC 6.x GPUs (Pascal) will use this code
NVCCFLAGS += --generate-code arch=compute_61,code=sm_61 # CC 6.1 GPUs
NVCCFLAGS += --generate-code arch=compute_62,code=sm_62 # CC 6.2 GPUs
NVCCFLAGS += --generate-code arch=compute_70,code=sm_70 # CC 7.x GPUs (Volta and Turing) will use this code
NVCCFLAGS += --generate-code arch=compute_72,code=sm_72 # CC 7.2 GPUs
NVCCFLAGS += --generate-code arch=compute_75,code=sm_75 # CC 7.5 GPUs (Turing) will use this code (slightly faster than 7.0)
NVCCFLAGS += --generate-code arch=compute_80,code=sm_80 # CC 8.0 GPUs (Ampere)
NVCCFLAGS += --generate-code arch=compute_86,code=sm_86 # CC 8.6 GPUs (Ampere)
NVCCFLAGS += --generate-code arch=compute_87,code=sm_87 # CC 8.7 GPUs
NVCCFLAGS += --generate-code arch=compute_89,code=sm_89 # CC 8.9 GPUs
NVCCFLAGS += --generate-code arch=compute_90,code=sm_90 # CC 9.0 GPUs (Hopper)

# pass some options to the C host compiler (e.g. gcc on Linux)
NVCCFLAGS += --compiler-options=-Wall

# Linker
LD = gcc
LDFLAGS = -fPIC $(CUDA_LIB) -lcudart_static -lm -lstdc++

##############################################################################

CSRC  = sieve.c timer.c parse.c read_config.c mfaktc.c checkpoint.c \
	signal_handler.c output.c crc.c
CUSRC = tf_96bit.cu tf_barrett96.cu tf_barrett96_gs.cu gpusieve.cu \
        cuda_basic_stuff.cu

COBJS  = $(CSRC:.c=.o)
CUOBJS = $(CUSRC:.cu=.o) tf_75bit.o

##############################################################################

all: ../mfaktc

../mfaktc : $(COBJS) $(CUOBJS)
	$(LD) $^ -o $@ $(LDFLAGS)

clean :
	rm -f *.o *~

sieve.o : sieve.c
	$(CC) $(CFLAGS) $(CFLAGS_EXTRA_SIEVE) -c $< -o $@
	
tf_75bit.o : tf_96bit.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ -DSHORTCUT_75BIT
	
%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@

##############################################################################
#
# created by ./create_dependencies.sh
#

mfaktc.o: mfaktc.c selftest-data-mersenne.c selftest-data-wagstaff.c \
 params.h my_types.h compatibility.h sieve.h read_config.h parse.h \
 timer.h tf_96bit.h tf_barrett96.h checkpoint.h signal_handler.h output.h \
 gpusieve.h cuda_basic_stuff.h selftest-data-mersenne.c

checkpoint.o: checkpoint.c params.h crc.h

output.o: output.c params.h my_types.h output.h compatibility.h crc.h

parse.o: parse.c compatibility.h parse.h params.h

read_config.o: read_config.c params.h my_types.h

sieve.o: sieve.c params.h compatibility.h

signal_handler.o: signal_handler.c params.h my_types.h compatibility.h

timer.o: timer.c timer.h compatibility.h

tf_96bit.o: tf_96bit.cu params.h my_types.h compatibility.h \
 my_intrinsics.h sieve.h timer.h output.h tf_debug.h \
 tf_96bit_base_math.cu tf_96bit_helper.cu gpusieve_helper.cu tf_common.cu \
 tf_common_gs.cu gpusieve.h

tf_barrett96.o: tf_barrett96.cu params.h my_types.h compatibility.h \
 my_intrinsics.h sieve.h timer.h output.h tf_debug.h \
 tf_96bit_base_math.cu tf_96bit_helper.cu tf_barrett96_div.cu \
 tf_barrett96_core.cu tf_common.cu

tf_barrett96_gs.o: tf_barrett96_gs.cu params.h my_types.h compatibility.h \
 my_intrinsics.h sieve.h timer.h output.h tf_debug.h \
 tf_96bit_base_math.cu tf_96bit_helper.cu tf_barrett96_div.cu \
 tf_barrett96_core.cu gpusieve_helper.cu tf_common_gs.cu gpusieve.h

gpusieve.o: gpusieve.cu params.h my_types.h compatibility.h \
 my_intrinsics.h gpusieve.h

cuda_basic_stuff.o: cuda_basic_stuff.cu params.h my_types.h

crc.o: crc.c crc.h

tf_75bit.o: tf_96bit.cu params.h my_types.h compatibility.h \
 my_intrinsics.h sieve.h timer.h output.h tf_debug.h \
 tf_96bit_base_math.cu tf_96bit_helper.cu gpusieve_helper.cu tf_common.cu \
 tf_common_gs.cu gpusieve.h
