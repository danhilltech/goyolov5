package goyolov5

// #cgo !cuda LDFLAGS: -lstdc++ -ltorch -ltorch_cpu -lc10
// #cgo cuda LDFLAGS: -lstdc++ -ltorch -ltorch_cpu -lc10 -lcuda -lcudart -lcublas -lcudnn -lcaffe2_nvrtc -lnvrtc-builtins -lnvrtc -lc10_cuda  -Wl,--no-as-needed -ltorch_cuda
// #cgo CFLAGS: -I${SRCDIR} -O3 -Wall -Wno-unused-variable -Wno-deprecated-declarations -Wno-c++11-narrowing -g -Wno-sign-compare -Wno-unused-function
// #cgo CXXFLAGS: -std=c++17 -I${SRCDIR} -g -O3
// #cgo CFLAGS: -D_GLIBCXX_USE_CXX11_ABI=1
import "C"
