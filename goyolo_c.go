package goyolo

//#include "stdlib.h"
//#include "goyolo.h"
import "C"
import (
	"fmt"
	"unsafe"
)

type Ctensor = C.tensor
type Cmodule = C.module

func (yolov5 *YoloV5) atRunInfer(inputTensor *Tensor) ([]float32, int, error) {
	cdevice := *(*C.int)(unsafe.Pointer(&yolov5.device))
	csize := *(*C.int)(unsafe.Pointer(&yolov5.size))
	hlf := 0
	if yolov5.half {
		hlf = 1
	}
	chalf := *(*C.int)(unsafe.Pointer(&hlf))
	cdata := unsafe.Pointer(&inputTensor.Pix[0])

	out := make([]float32, NPreds*PredSize)

	cout := unsafe.Pointer(&out[0])

	C.infer(yolov5.model, cdevice, cdata, csize, chalf, cout)
	if err := TorchErr(); err != nil {
		return nil, 0, err
	}

	return out, 1, nil
}

func atGetCUDADeviceCount() (int, error) {
	res := C.cudaDeviceCount()
	if err := TorchErr(); err != nil {
		return -1, err
	}
	return int(res), nil
}

// size_t at_dim(tensor);
func atDim(t Ctensor) uint64 {
	result := C.at_dim(t)
	return *(*uint64)(unsafe.Pointer(&result))
}

// void at_shape(tensor, int64_t *);
func atShape(t Ctensor, ptr unsafe.Pointer) {
	c_ptr := (*C.long)(ptr)
	C.at_shape(t, c_ptr)
}

// tensor at_get(tensor, int index);
func atGet(ts Ctensor, index int) Ctensor {
	cindex := *(*C.int)(unsafe.Pointer(&index))
	return C.at_get(ts, cindex)
}

// void at_free(tensor);
func atFree(ts Ctensor) {
	C.at_free(ts)
}

func atmLoadOnDevice(path string, device int32) Cmodule {
	ptr := C.CString(path)
	defer C.free(unsafe.Pointer(ptr))
	cdevice := *(*C.int)(unsafe.Pointer(&device))
	return C.atm_load_on_device(ptr, cdevice)
}

func atmInitModule(m Cmodule, half bool) {
	if half {
		C.init_module_half(m)
	} else {
		C.init_module(m)
	}

}

// TorchErr checks and retrieves last error message from
// C `thread_local` if existing and frees up C memory the C pointer
// points to.
//
// NOTE: Go language atm does not have generic function something
// similar to `macro` in Rust language, does it? So we have to
// wrap this function to any Libtorch C function call to check error
// instead of doing the other way around.
// See Go2 proposal: https://github.com/golang/go/issues/32620
func TorchErr() error {
	cptr := (*C.char)(GetAndResetLastErr())
	errStr := ptrToString(cptr)
	if errStr != "" {
		return fmt.Errorf("Libtorch API Error: %v\n", errStr)
	}

	return nil
}

func GetAndResetLastErr() *C.char {
	return C.get_and_reset_last_err()
}

// ptrToString check C pointer for null. If not null, get value
// the pointer points to and frees up C memory. It is used for
// getting error message C pointer points to and clean up C memory.
//
// NOTE: C does not have exception design. C++ throws exception
// to stderr. This code to check stderr for any err message,
// if it exists, takes it and frees up C memory.
func ptrToString(cptr *C.char) string {
	var str string = ""

	if cptr != nil {
		str = C.GoString(cptr)
		C.free(unsafe.Pointer(cptr))
	}

	return str
}
