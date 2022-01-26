package goyolov5

import (
	"fmt"
	"image"
	"image/png"
	_ "image/png"
	"os"
	"path/filepath"
	"testing"
)

func kbToMb(kb uint64) uint64 {
	return uint64(float64(kb) / 1024.0)
}

func TestGetCUDADeviceCount(t *testing.T) {
	_, err := atGetCUDADeviceCount()
	if err != nil {
		t.Fatal(err)
	}
}

func testInfer(t *testing.T, path string, yolov5 *YoloV5, expectedCount int) {
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	input, _, err := image.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	tensor := NewTensorFromImage(input)

	outTensor := NewTensorFromImage(tensor)

	predictions, err := yolov5.Infer(tensor, 0.5, 0.4, outTensor)
	if err != nil {
		t.Fatal(err)
	}

	bname := filepath.Base(path)

	f2, err := os.Create(fmt.Sprintf("tests/outputs/%s_%s_classes.png", yolov5.modelName, bname))
	if err != nil {
		t.Fatal(err)
	}
	defer f2.Close()
	if err = png.Encode(f2, outTensor); err != nil {
		t.Fatal(err)
	}

	if len(predictions[0]) != expectedCount {
		t.Fatalf("expected %d predictions, got %d", expectedCount, len(predictions[0]))
	}
}

func TestLetterboxingResize(t *testing.T) {
	f, err := os.Open("tests/inputs/640x480.png")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	input, _, err := image.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	tensor := NewTensorFromImage(input)

	f2, err := os.Create("tests/outputs/640x480_load.png")
	if err != nil {
		t.Fatal(err)
	}
	defer f2.Close()
	if err = png.Encode(f2, tensor); err != nil {
		t.Fatal(err)
	}

	lb, extraX, extraY, err := tensor.ToSquareShape()
	if err != nil {
		t.Fatal(err)
	}

	if extraX != 0 {
		t.Fatalf("expected 0, got %d", extraX)
	}
	if extraY != 160 {
		t.Fatalf("expected 160, got %d", extraX)
	}

	f3, err := os.Create("tests/outputs/640x480_lb.png")
	if err != nil {
		t.Fatal(err)
	}
	defer f3.Close()
	if err = png.Encode(f3, lb); err != nil {
		t.Fatal(err)
	}

	rs, _, err := lb.Resize(400)
	if err != nil {
		t.Fatal(err)
	}

	f4, err := os.Create("tests/outputs/640x480_rs.png")
	if err != nil {
		t.Fatal(err)
	}
	defer f4.Close()
	if err = png.Encode(f4, rs); err != nil {
		t.Fatal(err)
	}

}

func BenchmarkToSquare(b *testing.B) {

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
		tensor.ToSquareShape()
	}
}

func BenchmarkPreProcess(b *testing.B) {

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
		_, _, _, _, err := yolov5.preProcess(tensor)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRunInfer(b *testing.B) {
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

	inputTensor, _, _, _, err := yolov5.preProcess(tensor)
	if err != nil {
		b.Fatal(err)
	}

	for i := 0; i < 10; i++ {
		_, _, err := yolov5.atRunInfer(inputTensor)
		if err != nil {
			b.Fatal(err)
		}
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _, err := yolov5.atRunInfer(inputTensor)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPostConfidence(b *testing.B) {
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
	inputTensor := NewTensorFromImage(input)

	tensorBatch, _, err := yolov5.atRunInfer(inputTensor)
	if err != nil {
		b.Fatal(err)
	}

	confidenceThreshold := float32(0.5)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := yolov5.postConfidence(tensorBatch, int(NClasses), int(NPreds), confidenceThreshold)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNMS(b *testing.B) {
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
	inputTensor := NewTensorFromImage(input)

	tensorBatch, _, err := yolov5.atRunInfer(inputTensor)
	if err != nil {
		b.Fatal(err)
	}

	confidenceThreshold := float32(0.5)

	bboxes, err := yolov5.postConfidence(tensorBatch, int(NClasses), int(NPreds), confidenceThreshold)
	if err != nil {
		b.Fatal(err)
	}

	nmsThreshold := 0.4
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := yolov5.postNMS(bboxes, nmsThreshold)
		if err != nil {
			b.Fatal(err)
		}
	}
}
