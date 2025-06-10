UNAME_SYSTEM := $(shell uname -s)

CXXFLAGS = -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include

ifeq ($(UNAME_SYSTEM), Linux)
    AMDAPP_PATH ?= /opt/AMDAPP
    CXXFLAGS += -I${AMDAPP_PATH}/include -fPIC
    CUDAPATH ?= /usr/local/cuda
    NVCC = ${CUDAPATH}/bin/nvcc
    NVCCFLAGS = --compiler-options="-fPIC"
    LDFLAGS = -shared -fvisibility=hidden -L${CUDAPATH}/lib64 -lcuda -lcudart_static
    BUNDLE_DIR = HueWarp.ofx.bundle/Contents/Linux-x86-64/
    CUDA_OBJ = CudaKernel.o
    METAL_OBJ = 
    OPENCL_OBJ = OpenCLKernel.o
else
    ARCH_FLAGS = -arch arm64 -arch x86_64
    CXXFLAGS += ${ARCH_FLAGS}
    LDFLAGS = -bundle -fvisibility=hidden -F/Library/Frameworks -framework Metal -framework AppKit
    LDFLAGS += ${ARCH_FLAGS}
    BUNDLE_DIR = HueWarp.ofx.bundle/Contents/MacOS/
    CUDA_OBJ = CudaKernel.o
    METAL_OBJ = MetalKernel.o
    OPENCL_OBJ = 
endif

# Remove OpenCLKernel.o from the dependencies
HueWarp.ofx: HueWarp.o ${CUDA_OBJ} $(METAL_OBJ) $(OPENCL_OBJ) ofxsCore.o ofxsImageEffect.o ofxsInteract.o ofxsLog.o ofxsMultiThread.o ofxsParams.o ofxsProperty.o ofxsPropertyValidation.o
	$(CXX) $^ -o $@ $(LDFLAGS)
	mkdir -p $(BUNDLE_DIR)
	cp HueWarp.ofx $(BUNDLE_DIR)

# Platform-specific CUDA compilation
ifeq ($(UNAME_SYSTEM), Linux)
CudaKernel.o: CudaKernel.cu
	${NVCC} -c $< $(NVCCFLAGS)
else
CudaKernel.o: CudaKernel.cu
	cp CudaKernel.cu CudaKernel_mac.cpp
	$(CXX) -c CudaKernel_mac.cpp -o CudaKernel.o $(CXXFLAGS) -D__APPLE__ -D__MACH__
	rm CudaKernel_mac.cpp
endif

MetalKernel.o: MetalKernel.mm
	$(CXX) -c $< $(CXXFLAGS)

# Comment out or remove the OpenCL compilation rule
# OpenCLKernel.o: OpenCLKernel.cpp
#	$(CXX) -c $< $(CXXFLAGS)

%.o: ../Support/Library/%.cpp
	$(CXX) -c $< $(CXXFLAGS)

clean:
	rm -f *.o *.ofx
	rm -fr HueWarp.ofx.bundle

install: HueWarp.ofx
	cp -fr HueWarp.ofx.bundle /Library/OFX/Plugins
