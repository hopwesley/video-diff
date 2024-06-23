package main

import (
	"fmt"
	"gocv.io/x/gocv"
	"math"
)

const (
	DescriptorParam_M = 2
	DescriptorParam_m = 4
)

func SimpleSpatial() {
	video, err := gocv.VideoCaptureFile(param.rawAFile)
	if err != nil {
		panic(err)
	}

	var frameA = gocv.NewMat()
	var frameB = gocv.NewMat()
	if ok := video.Read(&frameA); !ok || frameA.Empty() {
		fmt.Println("Error reading video")
		frameA.Close()
		return
	}
	if ok := video.Read(&frameB); !ok || frameB.Empty() {
		fmt.Println("Error reading video")
		frameB.Close()
		return
	}
	var grayFrameA = gocv.NewMat()
	var grayFrameB = gocv.NewMat()
	gocv.CvtColor(frameA, &grayFrameA, gocv.ColorRGBToGray)
	gocv.CvtColor(frameB, &grayFrameB, gocv.ColorRGBToGray)
	frameA.Close()
	frameB.Close()
	grayFloat, err := matToFloatArray(grayFrameA)
	if err != nil {
		panic(err)
	}
	saveMatAsImage(grayFrameA, "simple_gray")
	saveJson("tmp/simple_gray.json", grayFloat)

	gradX := gocv.NewMat()
	gradY := gocv.NewMat()
	gradT := gocv.NewMat()

	gocv.Sobel(grayFrameA, &gradX, gocv.MatTypeCV16S, 1, 0, 3, 1, 0, gocv.BorderDefault)
	gocv.Sobel(grayFrameA, &gradY, gocv.MatTypeCV16S, 0, 1, 3, 1, 0, gocv.BorderDefault)
	gocv.AbsDiff(grayFrameA, grayFrameB, &gradT)

	saveMatAsImage(gradX, "simple_grad_x")
	saveMatAsImage(gradY, "simple_grad_y")
	saveMatAsImage(gradT, "simple_grad_t")

	gradXFloat, err := matToFloatArray(gradX)
	if err != nil {
		panic(err)
	}
	saveJson("tmp/simple_grad_x.json", gradXFloat)
	gradYFloat, err := matToFloatArray(gradY)
	if err != nil {
		panic(err)
	}
	saveJson("tmp/simple_grad_y.json", gradYFloat)
	gradTFloat, err := matToFloatArray(gradT)
	if err != nil {
		panic(err)
	}
	saveJson("tmp/simple_grad_t.json", gradTFloat)
	gradX.Close()
	gradY.Close()
	gradT.Close()
	grayFrameA.Close()
	grayFrameB.Close()
}

func matToFloatArray(mat gocv.Mat) ([][]float64, error) {
	if mat.Empty() {
		return nil, fmt.Errorf("mat is empty")
	}

	rows := mat.Rows()
	cols := mat.Cols()
	matType := mat.Type()

	// 创建一个二维浮点数组来存储梯度数据
	floatArray := make([][]float64, rows)
	for i := range floatArray {
		floatArray[i] = make([]float64, cols)
	}

	// 遍历 mat 并根据类型将每个像素值存储在浮点数组中
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			var value float64
			switch matType {
			case gocv.MatTypeCV8U:
				value = float64(mat.GetUCharAt(i, j))
			case gocv.MatTypeCV16S:
				value = float64(mat.GetShortAt(i, j))
			case gocv.MatTypeCV32F:
				value = float64(mat.GetFloatAt(i, j))
			case gocv.MatTypeCV64F:
				value = mat.GetDoubleAt(i, j)
			default:
				return nil, fmt.Errorf("unsupported mat type: %v", matType)
			}
			floatArray[i][j] = value
		}
	}

	return floatArray, nil
}

type frameCallback func(w, h float64, a, b, x, y, t *gocv.Mat)

func read2FrameFromSameVideo(file string, callback frameCallback) {
	video, err := gocv.VideoCaptureFile(file)
	if err != nil {
		panic(err)
	}
	width := video.Get(gocv.VideoCaptureFrameWidth)
	height := video.Get(gocv.VideoCaptureFrameHeight)
	var frameA = gocv.NewMat()
	var frameB = gocv.NewMat()
	if ok := video.Read(&frameA); !ok || frameA.Empty() {
		fmt.Println("Error reading video")
		frameA.Close()
		return
	}
	if ok := video.Read(&frameB); !ok || frameB.Empty() {
		fmt.Println("Error reading video")
		frameB.Close()
		return
	}
	var grayFrameA = gocv.NewMat()
	var grayFrameB = gocv.NewMat()
	gocv.CvtColor(frameA, &grayFrameA, gocv.ColorRGBToGray)
	gocv.CvtColor(frameB, &grayFrameB, gocv.ColorRGBToGray)
	frameA.Close()
	frameB.Close()

	gradX := gocv.NewMat()
	gradY := gocv.NewMat()
	gradT := gocv.NewMat()

	gocv.Sobel(grayFrameA, &gradX, gocv.MatTypeCV16S, 1, 0, 3, 1, 0, gocv.BorderDefault)
	gocv.Sobel(grayFrameA, &gradY, gocv.MatTypeCV16S, 0, 1, 3, 1, 0, gocv.BorderDefault)
	gocv.AbsDiff(grayFrameA, grayFrameB, &gradT)

	if callback != nil {
		callback(width, height, &grayFrameA, &grayFrameB, &gradX, &gradY, &gradT)
	}

	gradX.Close()
	gradY.Close()
	gradT.Close()

	grayFrameA.Close()
	grayFrameB.Close()
}

func IosQuantizeGradient() {
	read2FrameFromSameVideo(param.rawAFile, func(w, h float64, a, b, x, y, t *gocv.Mat) {
		grayFloat, err := matToFloatArray(*a)
		if err != nil {
			panic(err)
		}
		saveMatAsImage(*a, "simple_gray")
		saveJson("tmp/simple_gray.json", grayFloat)
		__saveImg(*t, "tmp/ios/simple_grad_t.png")
		gradTFloat, err := matToFloatArray(*t)
		if err != nil {
			panic(err)
		}
		saveJson("tmp/ios/simple_grad_t.json", gradTFloat)

		qg := quantizeGradients2(x, y, t)
		//result := computeFrameVector(qg)
		saveJson("tmp/ios/simple_quantize.json", qg)
	})
}

type Histogram [10]float64

func (h *Histogram) Add(hg Histogram) {
	for i := 0; i < 10; i++ {
		h[i] += hg[i]
	}
}
func (h *Histogram) dotProduct(h2 Histogram) float64 {
	var sum float64
	for i := 0; i < 10; i++ {
		sum += h[i] * h2[i]
	}
	return sum
}

// 计算Histogram的L2范数
func (h *Histogram) l2Norm() float64 {
	var sum float64
	for i := 0; i < 10; i++ {
		sum += h[i] * h[i]
	}
	return math.Sqrt(sum)
}

func AverageGradientOfBlock(S_0 int) {
	blockSize := S_0 / DescriptorParam_M / DescriptorParam_m
	read2FrameFromSameVideo(param.rawAFile, func(w, h float64, a, b, x, y, t *gocv.Mat) {
		__saveImg(*x, "tmp/ios/cpu_gradientXBuffer.png")
		__saveImg(*y, "tmp/ios/cpu_gradientYBuffer.png")
		__saveImg(*t, "tmp/ios/cpu_gradientTBuffer.png")

		width, height := int(w), int(h)
		var numberOfX = (width + blockSize - 1) / blockSize
		var numberOfY = (height + blockSize - 1) / blockSize
		var blockGradient = make([][][10]float64, numberOfY)
		var frameHistogram Histogram
		for rowIdx := 0; rowIdx < numberOfY; rowIdx++ {
			blockGradient[rowIdx] = make([][10]float64, numberOfX)
			for colIdx := 0; colIdx < numberOfX; colIdx++ {
				hg := quantizeGradientOfBlock(rowIdx, colIdx, blockSize, width, height, x, y, t)
				blockGradient[rowIdx][colIdx] = hg
				frameHistogram.Add(hg)
			}
		}
		__histogramToImg(blockGradient, "tmp/ios/cpu_block_gradient_.png")
		saveJson(fmt.Sprintf("tmp/ios/cpu_block_gradient_%d.json", S_0), blockGradient)
		saveJson(fmt.Sprintf("tmp/ios/cpu_frame_histogram_%d.json", S_0), frameHistogram)
	})
}

func quantizeGradientOfBlock(rowIdx, colIdx, blockSize, width, height int, gradientX, gradientY, gradientT *gocv.Mat) (histogram Histogram) {
	var startX = colIdx * blockSize
	var startY = rowIdx * blockSize
	var endX = startX + blockSize
	if endX > width {
		endX = width
	}
	var endY = startY + blockSize
	if endY > height {
		endY = height
	}
	var (
		sumGradientX = 0.0
		sumGradientY = 0.0
		sumGradientT = 0.0
		count        = 0.0
	)

	for row := startY; row < endY; row++ {
		for col := startX; col < endX; col++ {
			sumGradientX += float64(gradientX.GetShortAt(row, col))
			sumGradientY += float64(gradientY.GetShortAt(row, col))
			sumGradientT += float64(gradientT.GetUCharAt(row, col))
			count++
		}
	}
	//fmt.Println("sumGradientX:", sumGradientX, "sumGradientY:", sumGradientY, "sumGradientT:", sumGradientT, "count:", count)
	if count == 0 {
		return
	}

	gradient := [3]float64{sumGradientX / count, sumGradientY / count, sumGradientT / count}
	gradientL2 := norm2Float(gradient[:])
	if gradientL2 == 0.0 {
		return
	}

	gradient[0] = gradient[0] / gradientL2
	gradient[1] = gradient[1] / gradientL2
	gradient[2] = gradient[2] / gradientL2
	for i := 0; i < 10; i++ {
		pi, pi10 := projectGradient(gradient, icosahedronCenterP[i]), projectGradient(gradient, icosahedronCenterP[i+10])
		onePos := math.Abs(pi)
		twoPos := math.Abs(pi10)
		var val = onePos + twoPos - threshold
		if val < 0 {
			val = 0
		}
		histogram[i] = val
	}
	pL2 := norm2Float(histogram[:])
	if pL2 == 0.0 {
		return
	}
	for i := 0; i < 10; i++ {
		histogram[i] = histogram[i] * gradientL2 / pL2
	}

	return
}

type WeightsInOfCenter struct {
	x       float64
	y       float64
	weights [][]float64
}

func calculateCenters(width, height, S float64) (centersOfDesc []WeightsInOfCenter) {
	numberXOfBlock := DescriptorParam_M * DescriptorParam_m
	numberYOfBlock := DescriptorParam_M * DescriptorParam_m
	blockSize := S / float64(numberXOfBlock)
	sigma := float64(S) / 2.0 // Standard deviation for Gaussian kernel
	for centerY := S / 2; centerY < height; centerY += blockSize {
		for centerX := S / 2; centerX < width; centerX += blockSize {
			if centerX+S/2 > width || centerY+S/2 > height {
				continue
			}
			point := Point{X: centerX, Y: centerY}

			centerPoint := WeightsInOfCenter{
				x:       centerX,
				y:       centerY,
				weights: make([][]float64, numberYOfBlock),
			}

			blockStartX := point.X - S/2 + blockSize/2
			blockStartY := point.Y - S/2 + blockSize/2

			centerPoint.weights = make([][]float64, numberYOfBlock)
			for row := 0; row < numberYOfBlock; row++ {
				centerPoint.weights[row] = make([]float64, numberXOfBlock)
				for col := 0; col < numberXOfBlock; col++ {
					blockCenter := Point{X: blockStartX + float64(col)*blockSize, Y: blockStartY + float64(row)*blockSize}
					centerPoint.weights[row][col] = blockCenter.GaussianKernel(point, sigma)
				}
			}
			centersOfDesc = append(centersOfDesc, centerPoint)
		}
	}
	return
}

func GradientOfCell(S_0 int) {
	blockSize := S_0 / DescriptorParam_M / DescriptorParam_m
	read2FrameFromSameVideo(param.rawAFile, func(w, h float64, a, b, x, y, t *gocv.Mat) {
		//_ = calculateCenters(w, h, float64(S_0))
		width, height := int(w), int(h)
		var numberOfX = (width + blockSize - 1) / blockSize
		var numberOfY = (height + blockSize - 1) / blockSize
		var blockGradient = make([][][10]float64, numberOfY)
		for rowIdx := 0; rowIdx < numberOfY; rowIdx++ {
			blockGradient[rowIdx] = make([][10]float64, numberOfX)
			for colIdx := 0; colIdx < numberOfX; colIdx++ {
				blockGradient[rowIdx][colIdx] = quantizeGradientOfBlock(rowIdx, colIdx, blockSize, width, height, x, y, t)
			}
		}
		saveJson(fmt.Sprintf("tmp/ios/cpu_block_gradien_%d.json", S_0), blockGradient)
	})
}

func readingAllFrameOfVideo(video *gocv.VideoCapture, callback frameCallback2) {
	var preGrayFrame *gocv.Mat = nil
	for {
		var frameA = gocv.NewMat()
		if ok := video.Read(&frameA); !ok || frameA.Empty() {
			fmt.Println("video reading finished:")
			if preGrayFrame != nil {
				preGrayFrame.Close()
			}
			break
		}

		var curFrame = gocv.NewMat()
		gocv.CvtColor(frameA, &curFrame, gocv.ColorRGBToGray)
		frameA.Close()
		if preGrayFrame == nil {
			preGrayFrame = &curFrame
			continue
		}

		gradX := gocv.NewMat()
		gradY := gocv.NewMat()
		gradT := gocv.NewMat()

		gocv.Sobel(curFrame, &gradX, gocv.MatTypeCV16S, 1, 0, 3, 1, 0, gocv.BorderDefault)
		gocv.Sobel(curFrame, &gradY, gocv.MatTypeCV16S, 0, 1, 3, 1, 0, gocv.BorderDefault)
		gocv.AbsDiff(curFrame, *preGrayFrame, &gradT)

		if callback != nil {
			callback(&curFrame, preGrayFrame, &gradX, &gradY, &gradT)
		}

		gradX.Close()
		gradY.Close()
		gradT.Close()

		preGrayFrame.Close()
		preGrayFrame = &curFrame
	}
}

type frameCallback2 func(a, b, x, y, t *gocv.Mat)

func FrameQForTimeAlign(file string, S_0 int) []Histogram {
	blockSize := S_0 / DescriptorParam_M / DescriptorParam_m
	var frameHistogram []Histogram
	video, err := gocv.VideoCaptureFile(file)
	if err != nil {
		panic(err)
	}
	w := video.Get(gocv.VideoCaptureFrameWidth)
	h := video.Get(gocv.VideoCaptureFrameHeight)
	frameCount := video.Get(gocv.VideoCaptureFrameCount)
	fmt.Printf("video info: file:%s, width:%.2f height:%.2f frames:%.2f\n", file, w, h, frameCount)

	width, height := int(w), int(h)
	var numberOfX = (width + blockSize - 1) / blockSize
	var numberOfY = (height + blockSize - 1) / blockSize

	var counter = 0
	readingAllFrameOfVideo(video, func(a, b, x, y, t *gocv.Mat) {
		fmt.Println("frame finished id=>", counter)
		counter++
		var hgOneFrame Histogram
		for rowIdx := 0; rowIdx < numberOfY; rowIdx++ {
			for colIdx := 0; colIdx < numberOfX; colIdx++ {
				var hg = quantizeGradientOfBlock(rowIdx, colIdx, blockSize, width, height, x, y, t)
				hgOneFrame.Add(hg)
			}
		}

		frameHistogram = append(frameHistogram, hgOneFrame)
	})

	saveJson(fmt.Sprintf("tmp/ios/cpu_frame_q_%s_%d.json", file, S_0), frameHistogram)
	return frameHistogram
}

func normalizedCrossCorrelation(A, B []Histogram, maxOffset int) (int, float64) {
	bestOffset := 0
	maxCorrelation := -1.0

	for offset := -maxOffset; offset <= maxOffset; offset++ {
		var sum float64
		count := 0

		for t := 0; t < len(A); t++ {
			bt := t + offset
			if bt >= 0 && bt < len(B) {
				dot := A[t].dotProduct(B[bt])
				normA := A[t].l2Norm()
				normB := B[bt].l2Norm()
				if normA != 0 && normB != 0 {
					sum += dot / (normA * normB)
					count++
				}
			}
		}

		if count > 0 {
			correlation := sum / float64(count)
			fmt.Printf("Offset: %d, Correlation: %f\n", offset, correlation)
			if correlation > maxCorrelation {
				maxCorrelation = correlation
				bestOffset = offset
			}
		}
	}

	return bestOffset, maxCorrelation
}

func calculateSegmentCorrelation(segmentA, segmentB []Histogram) float64 {
	var sum float64
	count := len(segmentA)

	for i := 0; i < count; i++ {
		dot := segmentA[i].dotProduct(segmentB[i])
		normA := segmentA[i].l2Norm()
		normB := segmentB[i].l2Norm()
		if normA != 0 && normB != 0 {
			sum += dot / (normA * normB)
		}
	}

	return sum / float64(count)
}

func findBestSegmentAlignment(A, B []Histogram, segmentLength int) (int, int, float64) {
	bestAIndex := 0
	bestBIndex := 0
	maxCorrelation := -1.0

	for i := 0; i <= len(A)-segmentLength; i++ {
		for j := 0; j <= len(B)-segmentLength; j++ {
			segmentA := A[i : i+segmentLength]
			segmentB := B[j : j+segmentLength]
			correlation := calculateSegmentCorrelation(segmentA, segmentB)
			if correlation > maxCorrelation {
				maxCorrelation = correlation
				bestAIndex = i
				bestBIndex = j
			}
		}
	}

	return bestAIndex, bestBIndex, maxCorrelation
}

func extractFrames(inputFile string, outputFile string, startFrame int, endFrame int) {
	video, err := gocv.VideoCaptureFile(inputFile)
	if err != nil {
		panic(err)
	}
	defer video.Close()

	fps := video.Get(gocv.VideoCaptureFPS)
	width := video.Get(gocv.VideoCaptureFrameWidth)
	height := video.Get(gocv.VideoCaptureFrameHeight)

	writer, err := gocv.VideoWriterFile(outputFile, "mp4v", fps, int(width), int(height), true)
	if err != nil {
		panic(err)
	}
	defer writer.Close()

	for frameNum := startFrame; frameNum <= endFrame; frameNum++ {
		video.Set(gocv.VideoCapturePosFrames, float64(frameNum))
		frame := gocv.NewMat()
		if ok := video.Read(&frame); !ok {
			fmt.Printf("Error reading frame %d from video file\n", frameNum)
			break
		}
		writer.Write(frame)
		frame.Close()
	}
}

func AlignFrame() {
	var AQ []Histogram
	var BQ []Histogram
	readJson("tmp/ios/cpu_frame_q_A.mp4_32.json", &AQ)
	readJson("tmp/ios/cpu_frame_q_B.mp4_32.json", &BQ)
	segmentLength := 20
	//bestOffset, maxCorrelation := normalizedCrossCorrelation(AQ, BQ, maxOffset)
	bestAIndex, bestBIndex, maxCorrelation := findBestSegmentAlignment(AQ, BQ, segmentLength)

	fmt.Printf("Best A Index: %d, Best B Index: %d, Max Correlation: %f\n", bestAIndex, bestBIndex, maxCorrelation)
	extractFrames("A.mp4", "tmp/ios/align_a.mp4", bestAIndex, bestAIndex+120)
	extractFrames("B.mp4", "tmp/ios/align_b.mp4", bestBIndex, bestBIndex+120)
}
