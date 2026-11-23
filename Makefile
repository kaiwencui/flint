CXX ?= g++
NVCC ?= nvcc
CUDA_PATH ?= /usr/local/cuda
PREFIX ?= $(HOME)/.local
UNAME_S := $(shell uname -s)
STRIP_FLAGS := --strip-all
ifeq ($(UNAME_S),Darwin)
    STRIP_FLAGS :=
endif

# Auto-detect nvcc if CUDA is not explicitly specified
ifeq ($(origin CUDA),undefined)
    HAS_NVCC := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo 1 || echo 0)
    CUDA := $(HAS_NVCC)
endif

CUDA_ARCH ?= sm_89
# Automatically promote sm_120 -> sm_120a to enable Blackwell hardware PTX mma.sync.kind::mxf4nvf4 FP4 Tensor Cores
NVCC_ARCH := $(if $(filter sm_120,$(CUDA_ARCH)),sm_120a,$(CUDA_ARCH))
SM120A_DEF := $(if $(filter sm_120 sm_120a,$(CUDA_ARCH)),-DFLINT_HAS_FP4_TC=1,)

CXX_STD ?= $(shell $(CXX) -std=c++23 -x c++ -E /dev/null >/dev/null 2>&1 && echo c++23 || echo c++20)
CXXFLAGS_BASE := -std=$(CXX_STD) -Wall -Wextra -Wpedantic -Wno-unused-parameter -pthread -Isrc $(SM120A_DEF)

ifeq ($(DEBUG),1)
    CXXFLAGS := $(CXXFLAGS_BASE) -O0 -g3 -DFLINT_DEBUG=1 -fsanitize=address,undefined
    LDFLAGS := -pthread -fsanitize=address,undefined
    NVCCFLAGS := -std=c++20 -G -g -arch=$(NVCC_ARCH) -Isrc $(SM120A_DEF) -Xcompiler "-pthread"
else
    CXXFLAGS := $(CXXFLAGS_BASE) -O3 -march=native -DNDEBUG -fno-rtti
    LDFLAGS := -pthread -O3 -march=native
    NVCCFLAGS := -std=c++20 -O3 -DNDEBUG -arch=$(NVCC_ARCH) -Isrc $(SM120A_DEF) \
        -Xptxas -v -Xcompiler "-O3 -pthread -fno-rtti"
endif

BUILD_DIR := build

CPP_SRCS := \
	src/tokenizer.cpp \
	src/weights.cpp \
	src/cpu_ref.cpp \
	src/kv_cache.cpp \
	src/speculative.cpp \
	src/inkling2_engine.cpp \
	src/inkling2_forward.cpp \
	src/engine.cpp \
	src/server.cpp \
	src/cli.cpp

CPP_OBJS := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SRCS))
MAIN_OBJ := $(BUILD_DIR)/main.o
TEST_OBJ := $(BUILD_DIR)/test_main.o
BENCH_OBJ := $(BUILD_DIR)/bench_engine.o

CU_SRCS := \
	src/kernels/rmsnorm.cu \
	src/kernels/rope_kv.cu \
	src/kernels/gemv_bf16.cu \
	src/kernels/gemv_w4a16.cu \
	src/kernels/gemv_fp8.cu \
	src/kernels/flash_decode.cu \
	src/kernels/sampling.cu \
	src/kernels/gemv_nvfp4_moe.cu \
	src/kernels/inkling2_core.cu \
	src/kernels/inkling2_attn.cu \
	src/kernels/inkling2_wht.cu \
	src/kernels/inkling2_mtp.cu \
	src/kernels/megakernel_w4a16.cu

ifeq ($(CUDA),1)
    CXXFLAGS += -DFLINT_CUDA=1 -I$(CUDA_PATH)/include
    NVCCFLAGS += -DFLINT_CUDA=1
    CU_OBJS := $(patsubst src/kernels/%.cu,$(BUILD_DIR)/cu_%.o,$(CU_SRCS))
    CUDA_LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcublasLt -lcuda
else
    CXXFLAGS += -DFLINT_CUDA=0
    CU_OBJS :=
    CUDA_LDFLAGS :=
endif

LIB_OBJS := $(CPP_OBJS) $(CU_OBJS)
DEPS := $(CPP_OBJS:.o=.d) $(MAIN_OBJ:.o=.d) $(TEST_OBJ:.o=.d) $(BENCH_OBJ:.o=.d)

.PHONY: all clean test bench strip install

all: flint flint-bench

flint: $(LIB_OBJS) $(MAIN_OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^ $(CUDA_LDFLAGS)

flint-test: $(LIB_OBJS) $(TEST_OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^ $(CUDA_LDFLAGS)

flint-bench: $(LIB_OBJS) $(BENCH_OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^ $(CUDA_LDFLAGS)

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -MMD -MP -c -o $@ $<

$(BUILD_DIR)/cu_%.o: src/kernels/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

$(BUILD_DIR)/test_main.o: tests/test_main.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -MMD -MP -c -o $@ $<

$(BUILD_DIR)/bench_engine.o: tests/bench_engine.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -MMD -MP -c -o $@ $<

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

test: flint-test
	./flint-test

strip: flint
	strip $(STRIP_FLAGS) flint

install: flint
	install -d "$(DESTDIR)$(PREFIX)/bin"
	install -m 0755 flint "$(DESTDIR)$(PREFIX)/bin/flint"

clean:
	rm -rf $(BUILD_DIR) flint flint-test flint-bench

-include $(DEPS)
