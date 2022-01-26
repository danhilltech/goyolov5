package goyolov5

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"log"
	"unsafe"
)

// From https://github.com/pixiv/go-libjpeg/blob/master/rgb/rgb.go

// Tensor represent image data which has RGB colors.
// Tensor is compatible with image.RGBA, but does not have alpha channel to reduce using memory.
type Tensor struct {
	// Pix holds the image's stream, in R, G, B order.
	Pix []uint8
	// Stride is the Pix stride (in bytes) between vertically adjacent pixels.
	Stride int
	// Rect is the image's bounds.
	Rect image.Rectangle

	ctensor Ctensor
}

// NewTensor allocates and returns RGB image
func NewTensor(r image.Rectangle) *Tensor {
	w, h := r.Dx(), r.Dy()
	return &Tensor{Pix: make([]uint8, 3*w*h), Stride: 3 * w, Rect: r}
}

func NewTensorFromImage(i image.Image) *Tensor {
	r := i.Bounds()
	w, h := r.Dx(), r.Dy()
	newTensor := &Tensor{Pix: make([]uint8, 3*w*h), Stride: 3 * w, Rect: r}

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			newTensor.Set(x, y, i.At(x, y))
		}
	}

	return newTensor
}

// ColorModel returns RGB color model.
func (p *Tensor) ColorModel() color.Model {
	return ColorModel
}

// Bounds implements image.Image.At
func (p *Tensor) Bounds() image.Rectangle {
	return p.Rect
}

// At implements image.Image.At
func (p *Tensor) At(x, y int) color.Color {
	return p.RGBAAt(x, y)
}

// RGBAAt returns the color of the pixel at (x, y) as RGBA.
func (p *Tensor) RGBAAt(x, y int) color.RGBA {
	if !(image.Point{x, y}.In(p.Rect)) {
		return color.RGBA{}
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)*3
	return color.RGBA{p.Pix[i+0], p.Pix[i+1], p.Pix[i+2], 0xFF}
}

// ColorModel is RGB color model instance
var ColorModel = color.ModelFunc(rgbModel)

func rgbModel(c color.Color) color.Color {
	if _, ok := c.(RGB); ok {
		return c
	}
	r, g, b, _ := c.RGBA()
	return RGB{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8)}
}

// RGB color
type RGB struct {
	R, G, B uint8
}

// RGBA implements Color.RGBA
func (c RGB) RGBA() (r, g, b, a uint32) {
	r = uint32(c.R)
	r |= r << 8
	g = uint32(c.G)
	g |= g << 8
	b = uint32(c.B)
	b |= b << 8
	a = uint32(0xFFFF)
	return
}

func (p *Tensor) ToSquareShape() (*Tensor, int, int, error) {
	sb := p.Bounds()

	w := float64(sb.Max.X)
	h := float64(sb.Max.Y)

	fast := false

	if w > h {
		h = w
		fast = true
	} else {
		w = h
	}

	db := image.Rect(0, 0, int(w), int(h))

	// Now Center
	sw := sb.Max.X
	sh := sb.Max.Y
	dw := db.Max.X
	dh := db.Max.Y
	dr := image.Rect(
		0,
		0,
		sw,
		sh)

	extraX := dw - sw
	extraY := dh - sh

	dst := NewTensor(db)

	// draw the src image onto dst
	if fast {
		// dst.Pix[0:len(p.Pix)] = p.Pix
		copy(dst.Pix, p.Pix)
	} else {
		draw.Draw(dst, dr, p, p.Bounds().Min, draw.Src)
	}
	return dst, extraX, extraY, nil
}

func (p *Tensor) Set(x, y int, c color.Color) {
	if !(image.Point{x, y}.In(p.Rect)) {
		return
	}
	i := p.PixOffset(x, y)
	c1 := color.RGBAModel.Convert(c).(color.RGBA)
	s := p.Pix[i : i+3 : i+3] // Small cap improves performance, see https://golang.org/issue/27857
	s[0] = c1.R
	s[1] = c1.G
	s[2] = c1.B
}

// PixOffset returns the index of the first element of Pix that corresponds to
// the pixel at (x, y).
func (p *Tensor) PixOffset(x, y int) int {
	return (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)*3
}

func (img *Tensor) Resize(targetSize int) (*Tensor, float64, error) {

	// Trivial case: return input image
	if int(targetSize) == img.Bounds().Dx() && int(targetSize) == img.Bounds().Dy() {
		return img, 1.0, nil
	}

	// Input image has no pixels
	if img.Bounds().Dx() <= 0 || img.Bounds().Dy() <= 0 {
		return img, 1.0, nil
	}

	if img.Bounds().Dx() != img.Bounds().Dy() {
		return nil, 0.0, fmt.Errorf("cannot resize a non-square tensor")
	}

	if targetSize > img.Bounds().Dx() {
		return nil, 0.0, fmt.Errorf("cannot upscale a tensor")
	}

	newTensor := NewTensor(image.Rect(0, 0, targetSize, targetSize))

	dr := newTensor.Bounds()
	adr := newTensor.Bounds()
	sr := img.Bounds()

	ratio := float64(sr.Dx()) / float64(dr.Dx())

	dw2 := uint64(dr.Dx()) * 2
	dh2 := uint64(dr.Dy()) * 2
	sw := uint64(sr.Dx())
	sh := uint64(sr.Dy())
	for dy := int32(adr.Min.Y); dy < int32(adr.Max.Y); dy++ {
		sy := (2*uint64(dy) + 1) * sh / dh2
		d := (dr.Min.Y+int(dy)-newTensor.Rect.Min.Y)*newTensor.Stride + (dr.Min.X+adr.Min.X-newTensor.Rect.Min.X)*3
		for dx := int32(adr.Min.X); dx < int32(adr.Max.X); dx, d = dx+1, d+3 {
			sx := (2*uint64(dx) + 1) * sw / dw2
			pi := (sr.Min.Y+int(sy)-img.Rect.Min.Y)*img.Stride + (sr.Min.X+int(sx)-img.Rect.Min.X)*3
			pr := uint32(img.Pix[pi+0]) * 0x101
			pg := uint32(img.Pix[pi+1]) * 0x101
			pb := uint32(img.Pix[pi+2]) * 0x101
			newTensor.Pix[d+0] = uint8(pr >> 8)
			newTensor.Pix[d+1] = uint8(pg >> 8)
			newTensor.Pix[d+2] = uint8(pb >> 8)
		}
	}
	return newTensor, ratio, nil
}

// HLine draws a horizontal line
func (img *Tensor) HLine(x1, y, x2 int, col color.Color) {
	for ; x1 <= x2; x1++ {
		img.Set(x1, y, col)
	}
}

// VLine draws a veritcal line
func (img *Tensor) VLine(x, y1, y2 int, col color.Color) {
	for ; y1 <= y2; y1++ {
		img.Set(x, y1, col)
	}
}

// Rect draws a rectangle utilizing HLine() and VLine()
func (img *Tensor) DrawRect(rect image.Rectangle, col color.Color) {
	img.HLine(rect.Min.X, rect.Min.Y, rect.Max.X, col)
	img.HLine(rect.Min.X, rect.Max.Y, rect.Max.X, col)
	img.VLine(rect.Min.X, rect.Min.Y, rect.Max.Y, col)
	img.VLine(rect.Max.X, rect.Min.Y, rect.Max.Y, col)
}

// Drop drops (frees) the tensor
func (ts *Tensor) Drop() error {
	atFree(ts.ctensor)
	if err := TorchErr(); err != nil {
		return err
	}

	return nil
}

// Numel returns the total number of elements stored in a tensor.
func (ts *Tensor) Numel() uint {
	shape, err := ts.Size()
	if err != nil {
		log.Fatal(err)
	}
	return uint(FlattenDim(shape))
}

func bytesToInt64s(buf []byte) []int64 {
	if len(buf) < 1<<16 {
		return (*[1 << 13]int64)(unsafe.Pointer(&buf[0]))[0 : len(buf)/8 : len(buf)/8]
	}
	l := len(buf)
	if l > 1<<32 { // only use the first 2^32 bytes
		l = (1 << 32) - 1
	}
	return (*[1 << 29]int64)(unsafe.Pointer(&buf[0]))[0 : l/8 : l/8]
}

// FlattenDim counts number of elements with given shape
func FlattenDim(shape []int64) int {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}

	return int(n)
}

// Make sure Image implements image.Image.
// See https://golang.org/doc/effective_go.html#blank_implements.
var _ image.Image = new(Tensor)
