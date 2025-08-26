# configure path to CUDA Toolkit
CUDA_DIR = /usr/local/cuda
CUDA_INCLUDE = -I$(CUDA_DIR)/include/
CUDA_LIB = -L$(CUDA_DIR)/lib64/

# compiler settings for .c files (CPU)
CC = gcc
CFLAGS = -Wall -Wextra -O3 -flto -malign-double -ffunction-sections -fdata-sections -Wl,--gc-sections $(CUDA_INCLUDE)
CFLAGS_EXTRA_SIEVE = -funroll-all-loops

# compiler settings for .cu files (GPU)
NVCC = nvcc
NVCCFLAGS = $(CUDA_INCLUDE) --ptxas-options=-v -O3 -Wno-deprecated-gpu-targets

# generate code for compute capabilities - see this table for supported
# versions: https://en.wikipedia.org/wiki/CUDA#GPUs_supported

# older deprecated versions - uncomment to enable
# NVCCFLAGS += --generate-code arch=compute_11,code=sm_11 # mfaktc cannot use 1.0 but supports 1.1 and above
# NVCCFLAGS += --generate-code arch=compute_20,code=sm_20 # Fermi GPUs
# NVCCFLAGS += --generate-code arch=compute_30,code=sm_30 # early Kepler GPUs
# NVCCFLAGS += --generate-code arch=compute_35,code=sm_35 # later Kepler GPUs; funnel shift added in 3.2

# removed in CUDA 13.x - uncomment to enable
# NVCCFLAGS += --generate-code arch=compute_50,code=sm_50 # Maxwell GPUs
# NVCCFLAGS += --generate-code arch=compute_60,code=sm_60
# NVCCFLAGS += --generate-code arch=compute_61,code=sm_61
# NVCCFLAGS += --generate-code arch=compute_62,code=sm_62
# NVCCFLAGS += --generate-code arch=compute_70,code=sm_70
# NVCCFLAGS += --generate-code arch=compute_72,code=sm_72

# currently supported
NVCCFLAGS += --generate-code arch=compute_75,code=sm_75
NVCCFLAGS += --generate-code arch=compute_80,code=sm_80
NVCCFLAGS += --generate-code arch=compute_86,code=sm_86
NVCCFLAGS += --generate-code arch=compute_87,code=sm_87
NVCCFLAGS += --generate-code arch=compute_89,code=sm_89
NVCCFLAGS += --generate-code arch=compute_90,code=sm_90
NVCCFLAGS += --generate-code arch=compute_120,code=sm_120

# pass some options to the C host compiler
NVCCFLAGS += --compiler-options=-Wall

# Linker
LD = gcc
LDFLAGS = -flto -fPIC -Wl,--gc-sections $(CUDA_LIB) -lcudart_static -lm -lstdc++

INSTALL = install

##############################################################################

CSRC  = sieve.c timer.c parse.c read_config.c mfaktc.c checkpoint.c \
	signal_handler.c output.c
CUSRC = tf_72bit.cu tf_96bit.cu tf_barrett96.cu tf_barrett96_gs.cu gpusieve.cu

COBJS  = $(CSRC:.c=.o)
CUOBJS = $(CUSRC:.cu=.o) tf_75bit.o

##############################################################################

all: ../mfaktc ../mfaktc.ini

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

../%.ini : %.ini
	$(INSTALL) -m 644 $< ..

.PHONY: all clean
