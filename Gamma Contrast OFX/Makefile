UNAME_SYSTEM := $(shell uname -s)

CXXFLAGS = -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include

# Define all bundle directories for universal bundle
MAC_BUNDLE_DIR = ContrastSatVolume.ofx.bundle/Contents/MacOS/
WIN_BUNDLE_DIR = ContrastSatVolume.ofx.bundle/Contents/win64/
LINUX_BUNDLE_DIR = ContrastSatVolume.ofx.bundle/Contents/Linux-x86-64/

ifeq ($(UNAME_SYSTEM), Linux)
    AMDAPP_PATH ?= /opt/AMDAPP
    CXXFLAGS += -I${AMDAPP_PATH}/include -fPIC
    CUDAPATH ?= /usr/local/cuda
    NVCC = ${CUDAPATH}/bin/nvcc
    NVCCFLAGS = --compiler-options="-fPIC"
    LDFLAGS = -shared -fvisibility=hidden -L${CUDAPATH}/lib64 -lcuda -lcudart_static
    BUNDLE_DIR = $(LINUX_BUNDLE_DIR)
    CUDA_OBJ = CudaKernel.o
    METAL_OBJ = 
else ifeq ($(OS), Windows_NT)
    # Windows build settings
    CUDAPATH ?= "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
    NVCC = $(CUDAPATH)\bin\nvcc.exe
    NVCCFLAGS = -O2 -arch=sm_50
    CXXFLAGS += -I$(CUDAPATH)\include -DWIN32 -D_WIN32
    LDFLAGS = -shared -L$(CUDAPATH)\lib\x64 -lcuda -lcudart
    BUNDLE_DIR = $(WIN_BUNDLE_DIR)
    CUDA_OBJ = CudaKernel.o
    METAL_OBJ = 
    EXE_EXT = .exe
else
    # Mac build settings
    ARCH_FLAGS = -arch arm64 -arch x86_64
    CXXFLAGS += ${ARCH_FLAGS}
    LDFLAGS = -bundle -fvisibility=hidden -F/Library/Frameworks -framework OpenCL -framework Metal -framework AppKit
    LDFLAGS += ${ARCH_FLAGS}
    BUNDLE_DIR = $(MAC_BUNDLE_DIR)
    CUDA_OBJ = CudaKernel.o
    METAL_OBJ = MetalKernel.o
endif

# Create universal bundle structure
create-bundle-structure:
	mkdir -p $(MAC_BUNDLE_DIR)
	mkdir -p $(WIN_BUNDLE_DIR)
	mkdir -p $(LINUX_BUNDLE_DIR)
	mkdir -p ContrastSatVolume.ofx.bundle/Contents/Resources

# Mac binary target
mac-binary: create-bundle-structure ContrastSatVolume.o CudaKernel.o MetalKernel.o OpenCLKernel.o ofxsCore.o ofxsImageEffect.o ofxsInteract.o ofxsLog.o ofxsMultiThread.o ofxsParams.o ofxsProperty.o ofxsPropertyValidation.o
	$(CXX) ContrastSatVolume.o CudaKernel.o MetalKernel.o OpenCLKernel.o ofxsCore.o ofxsImageEffect.o ofxsInteract.o ofxsLog.o ofxsMultiThread.o ofxsParams.o ofxsProperty.o ofxsPropertyValidation.o -o ContrastSatVolume.ofx $(LDFLAGS)
	cp ContrastSatVolume.ofx $(MAC_BUNDLE_DIR)

# Windows cross-compilation (using MinGW or similar)
windows-binary: create-bundle-structure
	@echo "Building Windows binary using cross-compilation..."
	# Clean previous Windows objects
	rm -f *_win.o
	# Compile for Windows
	x86_64-w64-mingw32-g++ -c ContrastSatVolume.cpp -o ContrastSatVolume_win.o -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include -DWIN32 -D_WIN32
	x86_64-w64-mingw32-g++ -c OpenCLKernel.cpp -o OpenCLKernel_win.o -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include -DWIN32 -D_WIN32
	# Compile Support library objects for Windows
	x86_64-w64-mingw32-g++ -c ../Support/Library/ofxsCore.cpp -o ofxsCore_win.o -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include -DWIN32 -D_WIN32
	x86_64-w64-mingw32-g++ -c ../Support/Library/ofxsImageEffect.cpp -o ofxsImageEffect_win.o -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include -DWIN32 -D_WIN32
	x86_64-w64-mingw32-g++ -c ../Support/Library/ofxsInteract.cpp -o ofxsInteract_win.o -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include -DWIN32 -D_WIN32
	x86_64-w64-mingw32-g++ -c ../Support/Library/ofxsLog.cpp -o ofxsLog_win.o -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include -DWIN32 -D_WIN32
	x86_64-w64-mingw32-g++ -c ../Support/Library/ofxsMultiThread.cpp -o ofxsMultiThread_win.o -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include -DWIN32 -D_WIN32
	x86_64-w64-mingw32-g++ -c ../Support/Library/ofxsParams.cpp -o ofxsParams_win.o -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include -DWIN32 -D_WIN32
	x86_64-w64-mingw32-g++ -c ../Support/Library/ofxsProperty.cpp -o ofxsProperty_win.o -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include -DWIN32 -D_WIN32
	x86_64-w64-mingw32-g++ -c ../Support/Library/ofxsPropertyValidation.cpp -o ofxsPropertyValidation_win.o -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include -DWIN32 -D_WIN32
	# Create CPU fallback CUDA for Windows
	cp CudaKernel.cu CudaKernel_win.cpp
	sed -i 's/__global__//g' CudaKernel_win.cpp
	sed -i 's/blockIdx\.x \* blockDim\.x + threadIdx\.x/x/g' CudaKernel_win.cpp
	sed -i 's/blockIdx\.y \* blockDim\.y + threadIdx\.y/y/g' CudaKernel_win.cpp
	sed -i 's/fmaxf/std::max/g' CudaKernel_win.cpp
	sed -i 's/fminf/std::min/g' CudaKernel_win.cpp
	sed -i 's/powf/std::pow/g' CudaKernel_win.cpp
	echo '#include <algorithm>' | cat - CudaKernel_win.cpp > temp && mv temp CudaKernel_win.cpp
	echo '#include <cmath>' | cat - CudaKernel_win.cpp > temp && mv temp CudaKernel_win.cpp
	x86_64-w64-mingw32-g++ -c CudaKernel_win.cpp -o CudaKernel_win.o -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include -DWIN32 -D_WIN32
	# Link Windows binary
	x86_64-w64-mingw32-g++ ContrastSatVolume_win.o CudaKernel_win.o OpenCLKernel_win.o ofxsCore_win.o ofxsImageEffect_win.o ofxsInteract_win.o ofxsLog_win.o ofxsMultiThread_win.o ofxsParams_win.o ofxsProperty_win.o ofxsPropertyValidation_win.o -o ContrastSatVolume_win.ofx -shared -fvisibility=hidden
	cp ContrastSatVolume_win.ofx $(WIN_BUNDLE_DIR)ContrastSatVolume.ofx
	# Clean up temporary files
	rm -f *_win.o *_win.cpp ContrastSatVolume_win.ofx

# Linux cross-compilation (or native if tools available)
linux-binary: create-bundle-structure
	@echo "Building Linux binary using cross-compilation..."
	# Clean previous Linux objects
	rm -f *_linux.o
	# Check if we have cross-compilation tools
	@if command -v x86_64-linux-gnu-g++ >/dev/null 2>&1; then \
		echo "Using cross-compilation tools..."; \
		$(MAKE) linux-cross-compile; \
	else \
		echo "No cross-compilation tools found. Creating placeholder..."; \
		echo "#!/bin/bash" > $(LINUX_BUNDLE_DIR)ContrastSatVolume.ofx; \
		echo "echo 'Linux binary placeholder - build on Linux system'" >> $(LINUX_BUNDLE_DIR)ContrastSatVolume.ofx; \
		chmod +x $(LINUX_BUNDLE_DIR)ContrastSatVolume.ofx; \
	fi

linux-cross-compile:
	# Compile for Linux
	x86_64-linux-gnu-g++ -c ContrastSatVolume.cpp -o ContrastSatVolume_linux.o -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include -fPIC
	x86_64-linux-gnu-g++ -c OpenCLKernel.cpp -o OpenCLKernel_linux.o -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include -fPIC
	# Compile Support library objects for Linux
	x86_64-linux-gnu-g++ -c ../Support/Library/ofxsCore.cpp -o ofxsCore_linux.o -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include -fPIC
	x86_64-linux-gnu-g++ -c ../Support/Library/ofxsImageEffect.cpp -o ofxsImageEffect_linux.o -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include -fPIC
	x86_64-linux-gnu-g++ -c ../Support/Library/ofxsInteract.cpp -o ofxsInteract_linux.o -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include -fPIC
	x86_64-linux-gnu-g++ -c ../Support/Library/ofxsLog.cpp -o ofxsLog_linux.o -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include -fPIC
	x86_64-linux-gnu-g++ -c ../Support/Library/ofxsMultiThread.cpp -o ofxsMultiThread_linux.o -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include -fPIC
	x86_64-linux-gnu-g++ -c ../Support/Library/ofxsParams.cpp -o ofxsParams_linux.o -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include -fPIC
	x86_64-linux-gnu-g++ -c ../Support/Library/ofxsProperty.cpp -o ofxsProperty_linux.o -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include -fPIC
	x86_64-linux-gnu-g++ -c ../Support/Library/ofxsPropertyValidation.cpp -o ofxsPropertyValidation_linux.o -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include -fPIC
	# Create CPU fallback CUDA for Linux  
	cp CudaKernel.cu CudaKernel_linux.cpp
	sed -i 's/__global__//g' CudaKernel_linux.cpp
	sed -i 's/blockIdx\.x \* blockDim\.x + threadIdx\.x/x/g' CudaKernel_linux.cpp
	sed -i 's/blockIdx\.y \* blockDim\.y + threadIdx\.y/y/g' CudaKernel_linux.cpp
	sed -i 's/fmaxf/std::max/g' CudaKernel_linux.cpp
	sed -i 's/fminf/std::min/g' CudaKernel_linux.cpp
	sed -i 's/powf/std::pow/g' CudaKernel_linux.cpp
	echo '#include <algorithm>' | cat - CudaKernel_linux.cpp > temp && mv temp CudaKernel_linux.cpp
	echo '#include <cmath>' | cat - CudaKernel_linux.cpp > temp && mv temp CudaKernel_linux.cpp
	x86_64-linux-gnu-g++ -c CudaKernel_linux.cpp -o CudaKernel_linux.o -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include -fPIC
	# Link Linux binary
	x86_64-linux-gnu-g++ ContrastSatVolume_linux.o CudaKernel_linux.o OpenCLKernel_linux.o ofxsCore_linux.o ofxsImageEffect_linux.o ofxsInteract_linux.o ofxsLog_linux.o ofxsMultiThread_linux.o ofxsParams_linux.o ofxsProperty_linux.o ofxsPropertyValidation_linux.o -o ContrastSatVolume_linux.ofx -shared -fvisibility=hidden -fPIC
	cp ContrastSatVolume_linux.ofx $(LINUX_BUNDLE_DIR)ContrastSatVolume.ofx
	# Clean up temporary files
	rm -f *_linux.o *_linux.cpp ContrastSatVolume_linux.ofx

# Main target - builds for current platform only
ContrastSatVolume.ofx: create-bundle-structure ContrastSatVolume.o ${CUDA_OBJ} $(METAL_OBJ) OpenCLKernel.o ofxsCore.o ofxsImageEffect.o ofxsInteract.o ofxsLog.o ofxsMultiThread.o ofxsParams.o ofxsProperty.o ofxsPropertyValidation.o
	$(CXX) ContrastSatVolume.o ${CUDA_OBJ} $(METAL_OBJ) OpenCLKernel.o ofxsCore.o ofxsImageEffect.o ofxsInteract.o ofxsLog.o ofxsMultiThread.o ofxsParams.o ofxsProperty.o ofxsPropertyValidation.o -o ContrastSatVolume.ofx $(LDFLAGS)
	cp ContrastSatVolume.ofx$(EXE_EXT) $(BUNDLE_DIR)

# Build all platforms from Mac
all-platforms: mac-binary windows-binary linux-binary

# Create Info.plist for the bundle
create-info-plist:
	@echo "Creating Info.plist..."
	@echo '<?xml version="1.0" encoding="UTF-8"?>' > ContrastSatVolume.ofx.bundle/Contents/Info.plist
	@echo '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">' >> ContrastSatVolume.ofx.bundle/Contents/Info.plist
	@echo '<plist version="1.0">' >> ContrastSatVolume.ofx.bundle/Contents/Info.plist
	@echo '<dict>' >> ContrastSatVolume.ofx.bundle/Contents/Info.plist
	@echo '    <key>CFBundleExecutable</key>' >> ContrastSatVolume.ofx.bundle/Contents/Info.plist
	@echo '    <string>ContrastSatVolume.ofx</string>' >> ContrastSatVolume.ofx.bundle/Contents/Info.plist
	@echo '    <key>CFBundleIdentifier</key>' >> ContrastSatVolume.ofx.bundle/Contents/Info.plist
	@echo '    <string>com.OpenFXSample.ContrastSatVolume</string>' >> ContrastSatVolume.ofx.bundle/Contents/Info.plist
	@echo '    <key>CFBundleInfoDictionaryVersion</key>' >> ContrastSatVolume.ofx.bundle/Contents/Info.plist
	@echo '    <string>6.0</string>' >> ContrastSatVolume.ofx.bundle/Contents/Info.plist
	@echo '    <key>CFBundleName</key>' >> ContrastSatVolume.ofx.bundle/Contents/Info.plist
	@echo '    <string>ContrastSatVolume</string>' >> ContrastSatVolume.ofx.bundle/Contents/Info.plist
	@echo '    <key>CFBundlePackageType</key>' >> ContrastSatVolume.ofx.bundle/Contents/Info.plist
	@echo '    <string>BNDL</string>' >> ContrastSatVolume.ofx.bundle/Contents/Info.plist
	@echo '    <key>CFBundleVersion</key>' >> ContrastSatVolume.ofx.bundle/Contents/Info.plist
	@echo '    <string>1.0</string>' >> ContrastSatVolume.ofx.bundle/Contents/Info.plist
	@echo '</dict>' >> ContrastSatVolume.ofx.bundle/Contents/Info.plist
	@echo '</plist>' >> ContrastSatVolume.ofx.bundle/Contents/Info.plist

# Complete bundle with Info.plist and all platforms
bundle: all-platforms create-info-plist
	@echo "Universal OFX bundle created at: ContrastSatVolume.ofx.bundle"
	@echo "Bundle structure:"
	@find ContrastSatVolume.ofx.bundle -type f

# Platform-specific CUDA compilation
ifeq ($(UNAME_SYSTEM), Linux)
CudaKernel.o: CudaKernel.cu
	${NVCC} -c $< $(NVCCFLAGS)
else ifeq ($(OS), Windows_NT)
CudaKernel.o: CudaKernel.cu
	$(NVCC) -c $< $(NVCCFLAGS) -o $@
else
CudaKernel.o: CudaKernel.cu
	cp CudaKernel.cu CudaKernel_mac.cpp
	$(CXX) -c CudaKernel_mac.cpp -o CudaKernel.o $(CXXFLAGS) -D__APPLE__ -D__MACH__
	rm CudaKernel_mac.cpp
endif

MetalKernel.o: MetalKernel.mm
	$(CXX) -c $< $(CXXFLAGS)

%.o: ../Support/Library/%.cpp
	$(CXX) -c $< $(CXXFLAGS)

clean:
	rm -f *.o *.ofx *_win.* *_linux.*
	rm -fr ContrastSatVolume.ofx.bundle

install: bundle
ifeq ($(OS), Windows_NT)
	xcopy /E /I ContrastSatVolume.ofx.bundle "C:\Program Files\Common Files\OFX\Plugins\ContrastSatVolume.ofx.bundle"
else
	cp -fr ContrastSatVolume.ofx.bundle /Library/OFX/Plugins
endif

.PHONY: create-bundle-structure create-info-plist bundle all-platforms clean install mac-binary windows-binary linux-binary linux-cross-compile
