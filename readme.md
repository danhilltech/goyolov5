# YOLOV5 for golang

## Introduction


### Installation
[Libtorch](https://pytorch.org) needs to be availble to your C compiler and linker. For example:


```bash
cd /tmp
wget -O libtorch.zip https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.10.0%2Bcu102.zip
unzip libtorch.zip
mv /tmp/libtorch /usr/local/libtorch

export LIBRARY_PATH="/usr/local/libtorch/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="/usr/local/libtorch/lib:${LD_LIBRARY_PATH}"
export C_INCLUDE_PATH="/usr/local/libtorch/include:/usr/local/libtorch/include/torch/csrc/api/include:${C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="/usr/local/libtorch/include:/usr/local/libtorch/include/torch/csrc/api/include:${CPLUS_INCLUDE_PATH}"
```

### Basic example
This example loads a torchscript model exported from YOLOv5 and runs inference on the GPU. It also annotates the original image with bounding boxes around the detected classes.


```go
package main

import (
	"fmt"
	"image"
	"image/png"
	"os"

	"github.com/danhilltech/goyolov5"
)

func main() {
	yolov5, err := goyolov5.NewYoloV5("yolov5n.torchscript.gpu.batch1.pt", goyolov5.DeviceCPU, 640, false)
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
		panic(err)
	}
	tensor := goyolov5.NewTensorFromImage(input)

	outTensor := goyolov5.NewTensorFromImage(tensor)

	predictions, err := yolov5.Infer(tensor, 0.5, 0.4, outTensor)
	if err != nil {
		panic(err)
	}

	fmt.Println(predictions)

	f2, err := os.Create("path/to/my/annotated_image.png")
	if err != nil {
		panic(err)
	}
	defer f2.Close()
	if err = png.Encode(f2, outTensor); err != nil {
		panic(err)
	}
}

```

### CUDA
CUDA is supported, just build with the `cuda` tag. For example

```
go build --tags=cuda
```

The relevant CUDA libraries are assumed to be available to the linker. See the `.devcontainer/Dockerfile.amd64.cuda` as an example environment, which builds from the `nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04` image.


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