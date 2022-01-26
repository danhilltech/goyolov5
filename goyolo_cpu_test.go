//go:build !cuda

package goyolov5

import (
	"image"
	"os"
	"runtime"
	"testing"
)

func TestLoadInferCPUs(t *testing.T) {
	yolov5, err := NewYoloV5("weights/yolov5s/yolov5s.torchscript.cpu.640.pt", DeviceCPU, 640, false)
	if err != nil {
		t.Fatal(err)
	}

	testInfer(t, "tests/inputs/people.png", yolov5, 1)
	testInfer(t, "tests/inputs/people2.png", yolov5, 5)

}

func TestLoadInferCPUn(t *testing.T) {
	yolov5, err := NewYoloV5("weights/yolov5n/yolov5n.torchscript.cpu.640.pt", DeviceCPU, 640, false)
	if err != nil {
		t.Fatal(err)
	}

	testInfer(t, "tests/inputs/people.png", yolov5, 1)
	testInfer(t, "tests/inputs/people2.png", yolov5, 5)

}

func loadYoloForTesting(b *testing.B) *YoloV5 {
	yolov5, err := NewYoloV5("weights/yolov5n/yolov5n.torchscript.cpu.640.pt", DeviceCPU, 640, false)
	if err != nil {
		b.Fatal(err)
	}
	return yolov5
}

func BenchmarkInferCPU(b *testing.B) {
	yolov5 := loadYoloForTesting(b)

	f, err := os.Open("tests/inputs/640x480.png")
	if err != nil {
		b.Fatal(err)
	}
	defer f.Close()
	input, _, err := image.Decode(f)
	if err != nil {
		b.Fatal(err)
	}
	tensor := NewTensorFromImage(input)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := yolov5.Infer(tensor, 0.5, 0.4, nil)
		if err != nil {
			b.Fatal(err)
		}

	}
}
