//go:build cuda

package goyolo

import (
	"image"
	"os"
	"runtime"
	"testing"
)

func TestLoadInferGPUs(t *testing.T) {
	yolov5, err := NewYoloV5("weights/yolov5n/yolov5n.torchscript.gpu.batch1.pt", DeviceGPU, 640, false)
	if err != nil {
		t.Fatal(err)
	}

	testInfer(t, "tests/inputs/people.png", yolov5, 1)
	testInfer(t, "tests/inputs/people2.png", yolov5, 5)

}

func loadYoloForTesting(b *testing.B) *YoloV5 {
	yolov5, err := NewYoloV5("weights/yolov5n/yolov5n.torchscript.gpu.batch1.pt", DeviceGPU, 640, false)
	if err != nil {
		b.Fatal(err)
	}
	return yolov5
}
func BenchmarkInferGPU(b *testing.B) {
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

	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)

	for i := 0; i < 10; i++ {
		_, err := yolov5.Infer(tensor, 0.5, 0.4, nil)
		if err != nil {
			b.Fatal(err)
		}
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := yolov5.Infer(tensor, 0.5, 0.4, nil)
		if err != nil {
			b.Fatal(err)
		}

		if i%100 == 0 {
			runtime.ReadMemStats(&ms)
			b.Logf("Alloc: %f Mb, Total Alloc: %f MB, Sys: %f MB, Number of allocation: %d\n", float32(ms.Alloc)/float32(1024*1024), float32(ms.TotalAlloc)/float32(1024*1024), float32(ms.Sys)/float32(1024*1024), ms.HeapObjects)

		}
	}
}
