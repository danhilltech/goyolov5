package goyolov5

//#include "stdlib.h"
//#include "goyolo.h"
import "C"
import (
	"bytes"
	"encoding/binary"
	"fmt"
)

var nativeEndian binary.ByteOrder

func (ts *Tensor) Size() ([]int64, error) {
	dim := atDim(ts.ctensor)

	nbytes := int(8) * int(dim)

	dataPtr := C.malloc(C.size_t(nbytes))
	defer C.free(dataPtr)

	atShape(ts.ctensor, dataPtr)
	if err := TorchErr(); err != nil {
		return nil, err
	}

	dataSlice := (*[1 << 30]byte)(dataPtr)[:nbytes:nbytes]
	r := bytes.NewReader(dataSlice)

	out := make([]byte, nbytes)
	n, err := r.Read(out)
	if err != nil {
		return nil, err
	}
	if n != nbytes {
		return nil, fmt.Errorf("wrong read count %d vs %d", n, nbytes)
	}
	sz := bytesToInt64s(out)

	return sz, nil
}
