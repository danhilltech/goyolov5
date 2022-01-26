# YOLOV5 for golang

## Introduction

### Basic example
```go
package main

import (
	"github.com/danhilltech/goyolov5"
)

func main() {
	yolov5, err := goyolov5.NewYoloV5("yolov5n.torchscript.gpu.batch1.pt", DeviceGPU, 640, false)
	if err != nil {
		panic(err)
	}

	f, err := os.Open("path/to/my/image.png")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	input, _, err := image.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	tensor := goyolov5.NewTensorFromImage(input)

	outTensor := goyolov5.NewTensorFromImage(tensor)

	predictions, err := yolov5.Infer(tensor, 0.5, 0.4, outTensor)
	if err != nil {
		t.Fatal(err)
	}
}

```

### CUDA

### Weights
This library uses traced torchscript versions of YOLOv5. Instructions on exporting can be found in the YOLOv5 repository. Alternatively, there's a simple dockerfile plus script to generate CPU and GPU versions of the v6 release models:

```bash
make weights
```

## Developing
Inside the `.devcontainer` directory is an example `devcontainer.json` for use with VSCode. Duplicate the example and edit accordingly. Typically, this would mean adding or removing CUDA support depending on your hardware. E.g. add `--gpus all` to `runArgs` and `-tags=cuda` to gopls, testFlags, toolsEnvVars etc.

### Tests
Basic test coverage run

```
go test
```

or 

```
go test -tags=cuda
```

### CUDA
Make sure you've installed `nvidia-docker2` and the `nvidia-container-toolkit`.