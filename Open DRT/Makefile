UNAME_SYSTEM := $(shell uname -s)

CXXFLAGS = -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include

# Remove JSON library dependency as we use SimpleJSON
# CXXFLAGS += -ljsoncpp

ifeq ($(UNAME_SYSTEM), Linux)
    AMDAPP_PATH ?= /opt/AMDAPP
    CXXFLAGS += -I${AMDAPP_PATH}/include -fPIC
    CUDAPATH ?= /usr/local/cuda
    NVCC = ${CUDAPATH}/bin/nvcc
    NVCCFLAGS = --compiler-options="-fPIC"
    LDFLAGS = -shared -fvisibility=hidden -L${CUDAPATH}/lib64 -lcuda -lcudart_static
    BUNDLE_DIR = OpenDRT.ofx.bundle/Contents/Linux-x86-64/
    CUDA_OBJ = CudaKernel.o
    METAL_OBJ = 
    OPENCL_OBJ = OpenCLKernel.o
else
    ARCH_FLAGS = -arch arm64 -arch x86_64
    CXXFLAGS += ${ARCH_FLAGS}
    LDFLAGS = -bundle -fvisibility=hidden -F/Library/Frameworks -framework OpenCL -framework Metal -framework AppKit
    LDFLAGS += ${ARCH_FLAGS}
    BUNDLE_DIR = OpenDRT.ofx.bundle/Contents/MacOS/
    CUDA_OBJ = 
    METAL_OBJ = MetalKernel.o
    OPENCL_OBJ = 
endif

OpenDRT.ofx: OpenDRT.o MatrixManager.o SimpleJSON.o ${CUDA_OBJ} $(METAL_OBJ) $(OPENCL_OBJ) ofxsCore.o ofxsImageEffect.o ofxsInteract.o ofxsLog.o ofxsMultiThread.o ofxsParams.o ofxsProperty.o ofxsPropertyValidation.o
	$(CXX) $^ -o $@ $(LDFLAGS)
	mkdir -p $(BUNDLE_DIR)
	cp OpenDRT.ofx $(BUNDLE_DIR)
	cp ColorMatrices.json $(BUNDLE_DIR)

# Platform-specific CUDA compilation (Linux only)
ifeq ($(UNAME_SYSTEM), Linux)
CudaKernel.o: CudaKernel.cu
	${NVCC} -c $< $(NVCCFLAGS)

OpenCLKernel.o: OpenCLKernel.cpp
	$(CXX) -c $< $(CXXFLAGS)
endif

# macOS Metal compilation only
ifneq ($(UNAME_SYSTEM), Linux)
MetalKernel.o: MetalKernel.mm
	$(CXX) -c $< $(CXXFLAGS)
endif

MatrixManager.o: MatrixManager.cpp MatrixManager.h
	$(CXX) -c $< $(CXXFLAGS)

SimpleJSON.o: SimpleJSON.cpp SimpleJSON.h
	$(CXX) -c $< $(CXXFLAGS)

%.o: ../Support/Library/%.cpp
	$(CXX) -c $< $(CXXFLAGS)

clean:
	rm -f *.o *.ofx
	rm -fr OpenDRT.ofx.bundle

install: OpenDRT.ofx
	cp -fr OpenDRT.ofx.bundle /Library/OFX/Plugins
