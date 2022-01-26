package goyolov5

import (
	"image"
	"image/color"
	"math"
	"path/filepath"
	"sort"
)

type DeviceType = int32

const (
	DeviceCPU DeviceType = -1
	DeviceGPU DeviceType = 0
)

const (
	ClassesOffset = 5
	NPreds        = 25200
	PredSize      = 85
	NClasses      = PredSize - ClassesOffset
)

const (
	COCO_PERSON = 0
)

type YoloV5 struct {
	model     Cmodule
	device    DeviceType
	size      int
	modelName string
	half      bool
}

type Bbox struct {
	xmin            float64
	ymin            float64
	xmax            float64
	ymax            float64
	confidence      float64
	classIndex      uint
	classConfidence float64
}

type Prediction struct {
	Rect            image.Rectangle
	Confidence      float64
	ClassIndex      uint
	ClassConfidence float64
}

type ByConfBbox []Bbox

// Implement sort.Interface for []Bbox on Bbox.confidence:
// =====================================================
func (bb ByConfBbox) Len() int           { return len(bb) }
func (bb ByConfBbox) Less(i, j int) bool { return bb[i].confidence < bb[j].confidence }
func (bb ByConfBbox) Swap(i, j int)      { bb[i], bb[j] = bb[j], bb[i] }

// Intersection over union of two bounding boxes.
func Iou(b1, b2 Bbox) (retVal float64) {
	b1Area := (b1.xmax - b1.xmin + 1.0) * (b1.ymax - b1.ymin + 1.0)
	b2Area := (b2.xmax - b2.xmin + 1.0) * (b2.ymax - b2.ymin + 1.0)

	iXmin := math.Max(b1.xmin, b2.xmin)
	iXmax := math.Min(b1.xmax, b2.xmax)
	iYmin := math.Max(b1.ymin, b2.ymin)
	iYmax := math.Min(b1.ymax, b2.ymax)

	iArea := math.Max((iXmax-iXmin+1.0), 0.0) * math.Max((iYmax-iYmin+1.0), 0)

	return (iArea) / (b1Area + b2Area - iArea)
}

func DeviceCudaIfAvailable() DeviceType {
	cnt, _ := atGetCUDADeviceCount()
	if cnt > 0 {
		return DeviceGPU
	}
	return DeviceCPU
}

func NewYoloV5(path string, device DeviceType, size int, half bool) (*YoloV5, error) {

	module := atmLoadOnDevice(path, device)
	if err := TorchErr(); err != nil {
		return nil, err
	}

	atmInitModule(module, half)
	if err := TorchErr(); err != nil {
		return nil, err
	}

	yolov5 := &YoloV5{
		model:     module,
		device:    device,
		size:      size,
		modelName: filepath.Base(path),
		half:      half,
	}

	return yolov5, nil
}

func (yolov5 *YoloV5) preProcess(inputTensorRaw *Tensor) (*Tensor, int, int, float64, error) {
	inputTensorSquare, extraX, extraY, err := inputTensorRaw.ToSquareShape()
	if err != nil {
		return nil, 0, 0, 0.0, err
	}

	inputTensor, scaleRatio, err := inputTensorSquare.Resize(yolov5.size)
	if err != nil {
		return nil, 0, 0, 0.0, err
	}
	return inputTensor, extraX, extraY, scaleRatio, nil
}

func (yolov5 *YoloV5) postConfidence(tensor []float32, nclasses, npreds int, confidenceThreshold float32) ([][]Bbox, error) {
	var bboxes [][]Bbox = make([][]Bbox, int(nclasses))

	for index := 0; index < int(npreds); index++ {

		predVals := tensor[index*(nclasses+ClassesOffset) : (index+1)*(nclasses+ClassesOffset)]

		if predVals[4] > confidenceThreshold {
			classIndex := 0
			for i := 0; i < int(nclasses); i++ {
				if predVals[ClassesOffset+i] > predVals[ClassesOffset+classIndex] {
					classIndex = i
				}
			}

			if predVals[classIndex+5] > 0.0 {
				bbox := Bbox{
					xmin:            float64(predVals[0] - (predVals[2] / 2.0)),
					ymin:            float64(predVals[1] - (predVals[3] / 2.0)),
					xmax:            float64(predVals[0] + (predVals[2] / 2.0)),
					ymax:            float64(predVals[1] + (predVals[3] / 2.0)),
					confidence:      float64(predVals[4]),
					classIndex:      uint(classIndex),
					classConfidence: float64(predVals[5+classIndex]),
				}

				bboxes[classIndex] = append(bboxes[classIndex], bbox)
			}
		}
	}

	return bboxes, nil
}

func (yolov5 *YoloV5) postNMS(bboxes [][]Bbox, nmsThreshold float64) ([][]Bbox, error) {
	// Perform non-maximum suppression.
	var bboxesRes [][]Bbox
	for _, bboxesForClass := range bboxes {
		// 1. Sort by confidence
		sort.Sort(ByConfBbox(bboxesForClass))

		// 2.
		var currentIndex = 0
		for index := 0; index < len(bboxesForClass); index++ {
			drop := false
			for predIndex := 0; predIndex < currentIndex; predIndex++ {
				iou := Iou(bboxesForClass[predIndex], bboxesForClass[index])
				if iou > nmsThreshold {
					drop = true
					break
				}
			}

			if !drop {
				// swap
				bboxesForClass[currentIndex], bboxesForClass[index] = bboxesForClass[index], bboxesForClass[currentIndex]
				currentIndex += 1
			}
		}
		// 3. Truncate at currentIndex (exclusive)
		if currentIndex < len(bboxesForClass) {
			bboxesForClass = append(bboxesForClass[:currentIndex])
		}

		bboxesRes = append(bboxesRes, bboxesForClass)
	}

	return bboxesRes, nil
}

func (yolov5 *YoloV5) Infer(inputTensorRaw *Tensor, confidenceThreshold float32, nmsThreshold float64, annotateTensor *Tensor) ([][]Prediction, error) {

	// Preprocess
	inputTensor, _, _, scaleRatio, err := yolov5.preProcess(inputTensorRaw)
	if err != nil {
		return nil, err
	}

	// Infer
	rawOutput, batchSize, err := yolov5.atRunInfer(inputTensor)
	if err != nil {
		return nil, err
	}

	var outputPredictions [][]Prediction = make([][]Prediction, batchSize)

	for batch := 0; batch < batchSize; batch++ {

		tensorBatch := rawOutput[batch*NPreds : (batch+1)*NPreds]

		bboxes, err := yolov5.postConfidence(tensorBatch, NClasses, NPreds, confidenceThreshold)
		if err != nil {
			return nil, err
		}

		bboxesRes, err := yolov5.postNMS(bboxes, nmsThreshold)
		if err != nil {
			return nil, err
		}

		// Add to output
		// TODO Simplify this
		for _, c := range bboxesRes {
			for _, pred := range c {

				newPred := Prediction{
					Confidence:      pred.confidence,
					ClassIndex:      pred.classIndex,
					ClassConfidence: pred.classConfidence,
					Rect:            image.Rect(int(pred.xmin*scaleRatio), int(pred.ymin*scaleRatio), int(pred.xmax*scaleRatio), int(pred.ymax*scaleRatio)),
				}
				outputPredictions[batch] = append(outputPredictions[batch], newPred)

				// Draw on it
				if annotateTensor != nil && batchSize == 1 {
					annotateTensor.DrawRect(newPred.Rect, color.RGBA{0, 255, 0, 255})

				}
			}
		}

	}

	return outputPredictions, nil
}
