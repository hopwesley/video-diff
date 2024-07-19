package main

import (
	"fmt"
	"gocv.io/x/gocv"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"reflect"
	"time"
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

const HistogramSize = 10

type Histogram [HistogramSize]float64

func (h *Histogram) Add(hg Histogram) {
	for i := 0; i < HistogramSize; i++ {
		h[i] += hg[i]
	}
}

func (h *Histogram) Scale(s float64) (r Histogram) {
	for i := 0; i < HistogramSize; i++ {
		r[i] = h[i] * s
	}
	return
}

func (h *Histogram) dotProduct(h2 Histogram) float64 {
	var sum float64
	for i := 0; i < HistogramSize; i++ {
		sum += h[i] * h2[i]
	}
	return sum
}

// 计算Histogram的L2范数
func (h *Histogram) l2Norm() float64 {
	var sum float64
	for i := 0; i < HistogramSize; i++ {
		sum += h[i] * h[i]
	}
	return math.Sqrt(sum)
}

// 计算Histogram的L2范数
func (h *Histogram) length() float64 {
	var sum float64
	for i := 0; i < HistogramSize; i++ {
		sum += h[i] * h[i]
	}
	return sum
}

func (h *Histogram) mean() float64 {
	sum := 0.0
	for _, value := range h {
		sum += value
	}
	return sum / float64(HistogramSize)
}

func calculateDistance(h1, h2 Histogram) float64 {
	return h1.dotProduct(h2) / (h1.l2Norm() * h2.l2Norm())
}

func AverageGradientOfBlock(S_0 int, file string) {
	blockSize := S_0 / Cell_M / Cell_m
	read2FrameFromSameVideo(file, func(w, h float64, a, b, x, y, t *gocv.Mat) {
		grayFloat, _ := matToFloatArray(*x)
		saveJson("tmp/ios/cpu_gradientXBuffer.json", grayFloat)
		grayFloat, _ = matToFloatArray(*y)
		saveJson("tmp/ios/cpu_gradientYBuffer.json", grayFloat)
		grayFloat, _ = matToFloatArray(*t)
		saveJson("tmp/ios/cpu_gradientTBuffer.json", grayFloat)

		__saveImg(*x, "tmp/ios/cpu_gradientXBuffer.png")
		__saveImg(*y, "tmp/ios/cpu_gradientYBuffer.png")
		__saveImg(*t, "tmp/ios/cpu_gradientTBuffer.png")

		width, height := int(w), int(h)
		var numberOfX = (width + blockSize - 1) / blockSize
		var numberOfY = (height + blockSize - 1) / blockSize
		var blockGradient = make([][]Histogram, numberOfY)
		var frameHistogram Histogram
		for rowIdx := 0; rowIdx < numberOfY; rowIdx++ {
			blockGradient[rowIdx] = make([]Histogram, numberOfX)
			for colIdx := 0; colIdx < numberOfX; colIdx++ {
				hg := quantizeGradientOfBlock(rowIdx, colIdx, blockSize, width, height, x, y, t)
				blockGradient[rowIdx][colIdx] = hg
				frameHistogram.Add(hg)
			}
		}
		__histogramToImg(blockGradient, fmt.Sprintf("tmp/ios/cpu_block_gradient_%d.json.png", S_0))
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
	for i := 0; i < HistogramSize; i++ {
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
	numberXOfBlock := Cell_M * Cell_m
	numberYOfBlock := Cell_M * Cell_m
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

func testCpuOrGpu() {
	//var data [][][]float64
	//var result [10]float64
	//readJson("tmp/ios/gpu_frame_quantity_4_0.json", &data)
	//for i := 0; i < len(data); i++ {
	//	for j := 0; j < len(data[i]); j++ {
	//		for k := 0; k < 10; k++ {
	//			result[k] += data[i][j][k]
	//		}
	//	}
	//}
	//saveJson("tmp/ios/gpu_result_with_cpu.json", result)
	var grayA [][]uint8
	var grayB [][]uint8
	readJson("tmp/ios/gpu_grayBufferA_0.json", &grayA)
	readJson("tmp/ios/gpu_grayBufferB_0.json", &grayB)
	curFrame, _ := arrayToMat(grayB)
	preGrayFrame, _ := arrayToMat(grayA)
	gradX := gocv.NewMat()
	gradY := gocv.NewMat()
	gradT := gocv.NewMat()

	gocv.Sobel(curFrame, &gradX, gocv.MatTypeCV16S, 1, 0, 3, 1, 0, gocv.BorderDefault)
	gocv.Sobel(curFrame, &gradY, gocv.MatTypeCV16S, 0, 1, 3, 1, 0, gocv.BorderDefault)
	gocv.AbsDiff(curFrame, preGrayFrame, &gradT)

	grayFloat, _ := matToFloatArray(gradX)
	saveJson("tmp/ios/cpu_gpu_gradientXBuffer.json", grayFloat)
	grayFloat, _ = matToFloatArray(gradY)
	saveJson("tmp/ios/cpu_gpu_gradientYBuffer.json", grayFloat)
	grayFloat, _ = matToFloatArray(gradT)
	saveJson("tmp/ios/cpu_gpu_gradientTBuffer.json", grayFloat)

	width := len(grayA[0])
	height := len(grayA)
	blockSize := 32 / Cell_M / Cell_m
	var numberOfX = (width + blockSize - 1) / blockSize
	var numberOfY = (height + blockSize - 1) / blockSize
	var blockGradient = make([][][10]float64, numberOfY)
	var frameHistogram Histogram
	for rowIdx := 0; rowIdx < numberOfY; rowIdx++ {
		blockGradient[rowIdx] = make([][10]float64, numberOfX)
		for colIdx := 0; colIdx < 4; colIdx++ {
			hg := quantizeGradientOfBlock(rowIdx, colIdx, blockSize, width, height, &gradX, &gradY, &gradT)
			blockGradient[rowIdx][colIdx] = hg
			frameHistogram.Add(hg)
		}
		break
	}
	saveJson("tmp/ios/cpu_1_1.json", blockGradient)
	saveJson(fmt.Sprintf("tmp/ios/cpu_1_2.json"), frameHistogram)

	gradX.Close()
	gradY.Close()
	gradT.Close()

	preGrayFrame.Close()
	curFrame.Close()
}

func arrayToMat(array interface{}) (gocv.Mat, error) {
	// Ensure the input is a slice of slices
	v := reflect.ValueOf(array)
	if v.Kind() != reflect.Slice || v.Len() == 0 || v.Index(0).Kind() != reflect.Slice {
		return gocv.NewMat(), fmt.Errorf("input must be a non-empty slice of slices")
	}

	rows := v.Len()
	cols := v.Index(0).Len()
	if cols == 0 {
		return gocv.NewMat(), fmt.Errorf("input array has no columns")
	}

	var mat gocv.Mat
	var matType gocv.MatType

	// Determine the type of the elements and create the appropriate Mat
	elemType := v.Index(0).Index(0).Type()
	switch elemType.Kind() {
	case reflect.Float64:
		mat = gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV64F)
		matType = gocv.MatTypeCV64F
		break
	case reflect.Int16:
		mat = gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV16S)
		matType = gocv.MatTypeCV16S
		break
	case reflect.Uint8:
		mat = gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV8U)
		matType = gocv.MatTypeCV8U
		break
	default:
		return gocv.NewMat(), fmt.Errorf("unsupported element type: %v", elemType)
	}

	// Fill the Mat with the values from the input array
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			switch matType {
			case gocv.MatTypeCV64F:
				mat.SetDoubleAt(i, j, v.Index(i).Index(j).Float())
				break
			case gocv.MatTypeCV16S:
				mat.SetShortAt(i, j, int16(v.Index(i).Index(j).Int()))
				break
			case gocv.MatTypeCV8U:
				mat.SetUCharAt(i, j, uint8(v.Index(i).Index(j).Uint()))
				break
			default:
				panic("")
			}
		}
	}

	return mat, nil
}

func FrameQForTimeAlign(file string, S_0 int) []Histogram {
	blockSize := S_0 / Cell_M / Cell_m
	var frameHistogram []Histogram
	video, err := gocv.VideoCaptureFile(file)
	if err != nil {
		panic(err)
	}
	w := video.Get(gocv.VideoCaptureFrameWidth)
	h := video.Get(gocv.VideoCaptureFrameHeight)
	frameCount := video.Get(gocv.VideoCaptureFrameCount)

	width, height := int(w), int(h)
	var numberOfX = (width + blockSize - 1) / blockSize
	var numberOfY = (height + blockSize - 1) / blockSize
	fmt.Printf("video info: file:%s, width:%.2f height:%.2f frames:%.2f blockX:%d blockY:%d\n",
		file, w, h, frameCount, numberOfX, numberOfY)
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

	saveJson(fmt.Sprintf("tmp/ios/cpu_frame_histogram_%s_%d.json", file, S_0), frameHistogram)
	return frameHistogram
}

func AlignVideoFromStart() {
	AQ := FrameQForTimeAlign(param.rawAFile, 32)
	BQ := FrameQForTimeAlign(param.rawBFile, 32)

	ncc := nccOfAllFrameByHistogram(AQ, BQ)

	var gap = testTool.window

	startA, startB, _ := findMaxNCCSequence(ncc, gap)
	fmt.Printf("a=%d,b=%d gap=%d\n", startA, startB, gap)
	extractFrames(param.rawAFile, "tmp/ios/align_"+param.rawAFile+".mp4", startA, startA+gap)
	extractFrames(param.rawBFile, "tmp/ios/align_"+param.rawBFile+".mp4", startB, startB+gap)
}

func calculateNCCByHistogram(histogramA, histogramB Histogram) float64 {
	meanA := histogramA.mean() //calculateMean(histogramA)
	meanB := histogramB.mean()

	numerator := 0.0
	denominatorA := 0.0
	denominatorB := 0.0

	for i := 0; i < len(histogramA); i++ {
		numerator += (histogramA[i] - meanA) * (histogramB[i] - meanB)
		denominatorA += (histogramA[i] - meanA) * (histogramA[i] - meanA)
		denominatorB += (histogramB[i] - meanB) * (histogramB[i] - meanB)
	}

	return numerator / (math.Sqrt(denominatorA) * math.Sqrt(denominatorB))
}

const MiniNccVal = 0.9

func nccOfAllFrameByHistogram(aHisGramFloat, bHisGramFloat []Histogram) [][]float64 {

	videoALength := len(aHisGramFloat) // Video A frame count
	videoBLength := len(bHisGramFloat) // Video B frame count

	// Initialize a 2D array to store the NCC values
	nccValues := make([][]float64, videoALength)
	// Iterate over all frame pairs of Video A and Video B, calculate their NCC values
	for i, histogramA := range aHisGramFloat {
		nccValues[i] = make([]float64, videoBLength)
		for j, histogramB := range bHisGramFloat {
			var ncc = calculateNCCByHistogram(histogramA, histogramB)
			if MiniNccVal < ncc {
				nccValues[i][j] = ncc
			} else {
				nccValues[i][j] = 0.0
			}
			//nccValues[i][j] = calculateNCCByHistogram(histogramA, histogramB)
		}
	}
	return nccValues // These are the indices of the frames that best align in time
}

func alignTestA() {
	var AQ []Histogram
	var BQ []Histogram
	readJson("tmp/ios/cpu_frame_histogram_A_2.mp4_32.json", &AQ)
	readJson("tmp/ios/cpu_frame_histogram_B_2.mp4_32.json", &BQ)

	ncc := nccOfAllFrameByHistogram(AQ, BQ)

	saveJson("tmp/ios/ncc_a2_b2.json", ncc)
	//var Len = testTool.window

	//fmt.Println(findMinMaxCoordinates(ncc))
	gap, err := findMinMaxCoordinates(ncc)
	if err != nil {
		panic(err)
	}

	startA, startB, _ := findMaxNCCSequence(ncc, gap)
	fmt.Printf("a=%d,b=%d gap=%d\n", startA, startB, gap)

	extractFrames("A_2.mp4", "tmp/ios/align_a_2.mp4", startA, startA+gap)
	extractFrames("B_2.mp4", "tmp/ios/align_b_2.mp4", startB, startB+gap)
	//fmt.Println(findAlignmentPath(AQ, BQ, Len))
	//path := findAlignmentPath(AQ, BQ, Len)
	//cipherFrame("A_2.mp4", "B_2.mp4", "tmp/ios/", path)
}

func findMinMaxCoordinates(nccValues [][]float64) (gap int, err error) {
	minX, minY := math.MaxInt32, math.MaxInt32
	maxX, maxY := math.MinInt32, math.MinInt32
	found := false

	for i := 0; i < len(nccValues); i++ {
		for j := 0; j < len(nccValues[0]); j++ {
			if nccValues[i][j] != 0 {
				if i < minX {
					minX = i
				}
				if j < minY {
					minY = j
				}
				if i > maxX {
					maxX = i
				}
				if j > maxY {
					maxY = j
				}
				found = true
			}
		}
	}

	if !found {
		return -1, fmt.Errorf("no non-zero elements found")
	}
	return min(maxX-minX, maxY-minY), nil
}
func alignTestB() {
	var AQ []Histogram
	var BQ []Histogram
	//readJson("tmp/ios/cpu_frame_histogram_A.mp4_32.json", &AQ)
	//readJson("tmp/ios/cpu_frame_histogram_B.mp4_32.json", &BQ)
	readJson("tmp/ios/cpu_frame_histogram_align_a.mp4_32.json", &AQ)
	readJson("tmp/ios/cpu_frame_histogram_align_b.mp4_32.json", &BQ)

	ncc := nccOfAllFrameByHistogram(AQ, BQ)
	saveJson("tmp/ios/ncc_a_b.json", ncc)
	//var Len = testTool.window

	//fmt.Printf("a=%d,b=%d\n", startA, startB)
	gap, err := findMinMaxCoordinates(ncc)
	if err != nil {
		panic(err)
	}

	startA, startB, _ := findMaxNCCSequence(ncc, gap)
	fmt.Println("step1:", startA, startB, gap)
	extractFrames("A.mp4", "tmp/ios/align_a.mp4", startA, gap+startA)
	extractFrames("B.mp4", "tmp/ios/align_b.mp4", startB, gap+startB)
	//fmt.Println(findAlignmentPath(AQ, BQ, Len))
	//path := findAlignmentPath(AQ, BQ, Len)
	//cipherFrame("A.mp4", "B.mp4", "tmp/ios/", path)
}

func findMaxValueAndCoordinates(matrix [][]float64) (maxValue float64, maxX int, maxY int) {

	// 初始化最大值为负无穷，最大值坐标为-1
	maxValue = -math.MaxFloat64
	maxX, maxY = -1, -1

	// 遍历矩阵，寻找最大值及其坐标
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[0]); j++ {
			if matrix[i][j] > maxValue {
				maxValue = matrix[i][j]
				maxX = i
				maxY = j
			}
		}
	}

	return maxValue, maxX, maxY
}
func alignTestC() {
	var AQ []Histogram
	var BQ []Histogram
	err1, err2 := readJson("tmp/ios/gpu_frame_histogram_A.json", &AQ), readJson("tmp/ios/gpu_frame_histogram_B.json", &BQ)
	if err1 != nil || err2 != nil {
		fmt.Println(err1, err2)
		return
	}
	ncc := nccOfAllFrameByHistogram(AQ, BQ)
	saveJson("tmp/ios/cpu_gpu_ncc_a_b.json", ncc)
	var Len = testTool.window
	startA, startB, ws := findMaxNCCSequence(ncc, Len)
	fmt.Printf("normal a=%d,b=%d\n", startA, startB)
	fmt.Println(Len, "------>")
	fmt.Println(findMaxValueAndCoordinates(ws))
	fmt.Println("<------")
	fmt.Printf("summed a=%d,b=%d\n", startA, startB)
	extractFrames("A_2.mp4", "tmp/ios/gpu_align_a.mp4", startA, startA+Len)
	extractFrames("B_2.mp4", "tmp/ios/gpu_align_b.mp4", startB, startB+Len)
	saveJson("tmp/ios/cpu_gpu_ncc_weighted_sum.json", ws)

	var gpuNcc [][]float64
	readJson("tmp/ios/gpu_ncc_weighted_sum.json", &gpuNcc)
	_, startA, startB = findMaxValueAndCoordinates(gpuNcc)
	fmt.Printf("gpu new a=%d,b=%d\n", startA, startB)
	extractFrames("A_2.mp4", "tmp/ios/gpu_align_a_gpu.mp4", startA, startA+Len)
	extractFrames("B_2.mp4", "tmp/ios/gpu_align_b_gpu.mp4", startB, startB+Len)
}

func alignTestD() {
	var ncc [][]float64
	readJson("tmp/ios/gpu_ncc_a_b.json", &ncc)
	var Len = testTool.window
	startA, startB, ws := findMaxNCCSequence(ncc, Len)
	fmt.Printf("normal a=%d,b=%d\n", startA, startB)
	fmt.Println("------>")
	fmt.Println(findMaxValueAndCoordinates(ws))
	fmt.Println("<------")
	fmt.Printf("summed a=%d,b=%d\n", startA, startB)
	extractFrames("A_2.mp4", "tmp/ios/gpu_align_a.mp4", startA, startA+Len)
	extractFrames("B_2.mp4", "tmp/ios/gpu_align_b.mp4", startB, startB+Len)
	saveJson("tmp/ios/cpu_gpu_ncc_weighted_sum.json", ws)
}

func alignTestE() {
	var ws [][]float64
	var Len = testTool.window
	readJson("tmp/ios/gpu_ncc_weighted_sum.json", &ws)
	val, startA, startB := findMaxValueAndCoordinates(ws)
	fmt.Printf("summed val=%.5f a=%d,b=%d\n", val, startA, startB)
	extractFrames("A_2.mp4", "tmp/ios/gpu_align_a.mp4", startA, startA+Len)
	extractFrames("B_2.mp4", "tmp/ios/gpu_align_b.mp4", startB, startB+Len)
}
func alignTestF() {
	var AQ []Histogram
	var BQ []Histogram
	readJson("tmp/ios/cpu_frame_histogram_A.mp4_32.json", &AQ)
	readJson("tmp/ios/cpu_frame_histogram_B.mp4_32.json", &BQ)

	ncc := nccOfAllFrameByHistogram(AQ, BQ)
	saveJson("tmp/ios/ncc_a_b.json", ncc)
	var gap = testTool.window

	//fmt.Printf("a=%d,b=%d\n", startA, startB)
	//gap, err := findMinMaxCoordinates(ncc)
	//if err != nil {
	//	panic(err)
	//}

	startA, startB, _ := findMaxNCCSequence(ncc, gap)
	fmt.Println("step1:", startA, startB, gap)
	extractFrames("A.mp4", "tmp/ios/align_a.mp4", startA, gap+startA)
	extractFrames("B.mp4", "tmp/ios/align_b.mp4", startB, gap+startB)
	//fmt.Println(findAlignmentPath(AQ, BQ, Len))
	//path := findAlignmentPath(AQ, BQ, Len)
	//cipherFrame("A.mp4", "B.mp4", "tmp/ios/", path)

	var cipheredAQ = make([]Histogram, gap)
	var cipheredBQ = make([]Histogram, gap)
	for i := 0; i < gap; i++ {
		cipheredAQ[i] = AQ[startA+i]
		cipheredBQ[i] = BQ[startB+i]
	}

	saveJson("tmp/ios/ciphered_cpu_frame_histogram_A.json", cipheredAQ)
	saveJson("tmp/ios/ciphered_cpu_frame_histogram_B.json", cipheredBQ)
}
func compareTestA() {
	histogramToImg("tmp/ios/gpu_average_block_2_a_level_0.json")
	histogramToImg("tmp/ios/gpu_average_block_2_a_level_1.json")
	histogramToImg("tmp/ios/gpu_average_block_2_a_level_2.json")
	histogramToImg("tmp/ios/gpu_average_block_2_b_level_0.json")
	histogramToImg("tmp/ios/gpu_average_block_2_b_level_1.json")
	histogramToImg("tmp/ios/gpu_average_block_2_b_level_2.json")
	//var avgQG [][]Histogram
	//readJson("tmp/ios/gpu_average_block_4_2.json", &avgQG)
	//__histogramToImg(avgQG, "tmp/ios/gpu_average_block_4_2.json.png")
}
func compareTestB() {
	var normalizedDescriptor [][][]float64
	readJson("tmp/ios/gpu_descriptor_2_a_level_0.json", &normalizedDescriptor)
	__histogramToImgFloat(normalizedDescriptor, fmt.Sprintf("tmp/ios/gpu_descriptor_2_a_level_0.json.png"))

	readJson("tmp/ios/gpu_descriptor_2_a_level_1.json", &normalizedDescriptor)
	__histogramToImgFloat(normalizedDescriptor, fmt.Sprintf("tmp/ios/gpu_descriptor_2_a_level_1.json.png"))

	readJson("tmp/ios/gpu_descriptor_2_a_level_2.json", &normalizedDescriptor)
	__histogramToImgFloat(normalizedDescriptor, fmt.Sprintf("tmp/ios/gpu_descriptor_2_a_level_2.json.png"))
}

func compareTestC() {
	var wtl [][]float64
	readJson("tmp/ios/gpu_wtl_2_level_0.json", &wtl)
	__saveNormalizedData(normalizeImage(wtl), fmt.Sprintf("tmp/ios/gpu_wtl_2_level_0.json.png"))
	readJson("tmp/ios/gpu_wtl_2_level_1.json", &wtl)
	__saveNormalizedData(normalizeImage(wtl), fmt.Sprintf("tmp/ios/gpu_wtl_2_level_1.json.png"))
	readJson("tmp/ios/gpu_wtl_2_level_2.json", &wtl)
	__saveNormalizedData(normalizeImage(wtl), fmt.Sprintf("tmp/ios/gpu_wtl_2_level_2.json.png"))

}
func compareTestD() {
	var fullImg [][]float64
	err := readJson("tmp/ios/gpu_wtl_2_billinear_0.json", &fullImg)
	if err == nil {
		__saveNormalizedData(normalizeImage(fullImg), fmt.Sprintf("tmp/ios/gpu_wtl_2_billinear_0.json.png"))
	}
	err = readJson("tmp/ios/gpu_wtl_2_billinear_1.json", &fullImg)
	if err == nil {
		__saveNormalizedData(normalizeImage(fullImg), fmt.Sprintf("tmp/ios/gpu_wtl_2_billinear_1.json.png"))
	}
	err = readJson("tmp/ios/gpu_wtl_2_billinear_2.json", &fullImg)
	if err == nil {
		__saveNormalizedData(normalizeImage(fullImg), fmt.Sprintf("tmp/ios/gpu_wtl_2_billinear_2.json.png"))
	}

	err = readJson("tmp/ios/gpu_gradient_magnitude_2_.json", &fullImg)
	if err == nil {
		__saveNormalizedData(normalizeImage(fullImg), fmt.Sprintf("tmp/ios/gpu_gradient_magnitude_2_.json.png"))
	}
}

func compareTestE() {
	var fullImg [][]float64
	err := readJson("tmp/ios/gpu_wtl_2_billinear_final_.json", &fullImg)
	if err == nil {
		__saveNormalizedData(normalizeImage(fullImg), fmt.Sprintf("tmp/ios/gpu_wtl_2_billinear_final_.json.png"))
	}
	readJson("tmp/ios/gpu_wtl_2_billinear_final_normalized_.json", &fullImg)
	__saveNormalizedData(normalizeImage(fullImg), fmt.Sprintf("tmp/ios/gpu_wtl_2_billinear_final_normalized_.json.png"))

	err = readJson("tmp/ios/gpu_adjust_map_2_.json", &fullImg)
	if err == nil {
		__saveNormalizedData(normalizeImage(fullImg), fmt.Sprintf("tmp/ios/gpu_adjust_map_2_.json.png"))
	}
}

func compareTestF() {
	hist := make([][]int, 256)
	err := readJson("tmp/ios/gpu_percentile_2_histogram_.json", &hist)
	if err != nil {
		fmt.Println(err)
		return
	}

	total := 1280 * 720
	lowCount := int(float64(total) * 1 / 100)
	highCount := int(float64(total) * 99 / 100)
	lowVal, highVal := 0, 255
	cumulative := 0
	for i, count := range hist[0] {
		cumulative += count
		if cumulative <= lowCount {
			lowVal = i
		}
		if cumulative <= highCount {
			highVal = i
		}
	}

	fmt.Println("lowVal:", lowVal, "highVal:", highVal)
}

func compareTestG() {
	var frameA [][]uint8
	readJson("tmp/ios/gpu_gray_frameA_2_.json", &frameA)
	saveGrayDataToImg(frameA, "tmp/ios/gpu_gray_frameA_2_.json.png")

	var gradientMagnitude [][]float64
	readJson("tmp/ios/gpu_gradient_magnitude_2_.json", &gradientMagnitude)
	__saveNormalizedData(normalizeImage(gradientMagnitude), "tmp/ios/gpu_gradient_magnitude_2_.json.png")

	var normalizedMap [][]float64
	readJson("tmp/ios/gpu_wtl_2_billinear_final_normalized_.json", &normalizedMap)
	__saveNormalizedData(normalizeImage(normalizedMap), "tmp/ios/gpu_wtl_2_billinear_final_normalized_.json.png")

	// 预先分配 byteSlice 的容量
	rows := len(frameA)
	cols := len(frameA[0])
	byteSlice := make([]byte, 0, rows*cols)

	// 将 frameA 的数据追加到 byteSlice
	for _, row := range frameA {
		byteSlice = append(byteSlice, row...)
	}

	// 从字节数组创建 Mat
	mat, err := gocv.NewMatFromBytes(rows, cols, gocv.MatTypeCV8U, byteSlice)
	if err != nil {
		panic(err)
	}
	defer mat.Close()
	//__saveImg(mat, "tmp/ios/gpu_gray_frameA_2_2.json.png")

	var adjustedFrame [][]float64
	readJson("tmp/ios/gpu_adjust_map_2_.json", &adjustedFrame)
	__saveNormalizedData(adjustedFrame, "tmp/ios/gpu_adjust_map_2_.json.png")

	img := overlay2(mat, normalizedMap, gradientMagnitude)

	file, _ := os.Create(fmt.Sprintf("tmp/ios/gpu_one_frame_overlay.png"))
	_ = png.Encode(file, img)
	_ = file.Close()

	var gpuImgData [][][]float64
	readJson("tmp/ios/gpu_onverlay_image_raw_2_.json", &gpuImgData)

	var height = len(gpuImgData)
	var width = len(gpuImgData[0])
	gpuImg := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			vColor := color.RGBA{
				R: uint8(gpuImgData[y][x][0] * 255.0),
				G: uint8(gpuImgData[y][x][1] * 255.0),
				B: uint8(gpuImgData[y][x][2] * 255.0),
				A: uint8(gpuImgData[y][x][3] * 255.0),
			}
			gpuImg.Set(x, y, vColor)
		}
	}

	file, _ = os.Create(fmt.Sprintf("tmp/ios/gpu_onverlay_image_raw_2_.json.png"))
	_ = png.Encode(file, img)
	_ = file.Close()
}

func CommTest() {
	//compareTestA()
	//compareTestB()
	//compareTestC()
	//compareTestD()
	//compareTestE()
	//compareTestF()
	compareTestG()
	//var sigma = 1.0
	//alignTestA()
	//alignTestB()
	//alignTestC()
	//alignTestD()
	//alignTestE()
	//alignTestF()
	//av, _ := gocv.VideoCaptureFile("A_2.mp4")
	//bv, _ := gocv.VideoCaptureFile("B_2.mp4")
	//saveVideoFromFrame(av, startA, "tmp/ios/align_a.mp4")
	//saveVideoFromFrame(bv, startB, "tmp/ios/align_b.mp4")

	//var AQ []Histogram
	//var BQ []Histogram
	//readJson("tmp/ios/cpu_frame_histogram_A.mp4_32.json", &AQ)
	//readJson("tmp/ios/cpu_frame_histogram_B.mp4_32.json", &BQ)

	//readJson("tmp/ios/cpu_frame_histogram_A_2.mp4_32.json", &AQ)
	//readJson("tmp/ios/cpu_frame_histogram_B_2.mp4_32.json", &BQ)
	//segmentLength := 20
	//bestOffset, maxCorrelation := normalizedCrossCorrelation(AQ, BQ, maxOffset)
	//bestAIndex, bestBIndex, maxCorrelation := findBestSegmentAlignment(AQ, BQ, segmentLength)
	//var vLen = 200
	//bestOffsetA, bestOffsetB, maxCorr := normalizedCrossCorrelationWithWindow(AQ, BQ, vLen)
	//bestOffsetA, maxCorr := normalizedCrossCorrelation(AQ, BQ)

	//
	//fmt.Printf("Best A Index: %d,Best B Index: %d, Max Correlation: %f\n", bestOffsetA, bestOffsetB, maxCorr)
	//extractFrames("A_2.mp4", "tmp/ios/align_a.mp4", bestOffsetA, bestOffsetA+vLen)
	//extractFrames("B_2.mp4", "tmp/ios/align_b.mp4", bestOffsetB, bestOffsetB+vLen)

	//path, maxCorr := DTW(AQ, BQ)
	//fmt.Printf("Best A Index: %v, Max Correlation: %f\n", path, maxCorr)
	//cipherFrame("A.mp4", "B.mp4", "tmp/ios/", path)
	//cipherFrame("A.mp4", "B.mp4", "tmp/ios/", path)
	//_ = AlignVideos("A_2.mp4", "B_2.mp4", path, "tmp/ios/a_align.mp4", "tmp/ios/b_align.mp4")
}

func testZeroFrameGradient() {
	var blockAvgGradient [][]Histogram
	readJson("tmp/ios/gpu_frame_quantity_4_0.json", &blockAvgGradient)
	var frameHistogram []Histogram

	var hgOneFrame Histogram
	for rowIdx := 0; rowIdx < len(blockAvgGradient); rowIdx++ {
		for colIdx := 0; colIdx < len(blockAvgGradient[rowIdx]); colIdx++ {
			hgOneFrame.Add(blockAvgGradient[rowIdx][colIdx])
		}
	}
	frameHistogram = append(frameHistogram, hgOneFrame)
	saveJson("tmp/ios/cpu_gpu_frame_histogram.json", frameHistogram)
}

func localDTW(a, b []Histogram, window int) float64 {
	n, m := len(a), len(b)
	dtw := make([][]float64, n+1)
	for i := range dtw {
		dtw[i] = make([]float64, m+1)
		for j := range dtw[i] {
			dtw[i][j] = math.Inf(1)
		}
	}
	dtw[0][0] = 0
	for i := 1; i <= n; i++ {
		for j := max(1, i-window); j <= min(m, i+window); j++ {
			cost := calculateDistance(a[i-1], b[j-1])
			dtw[i][j] = cost + math.Min(dtw[i-1][j], math.Min(dtw[i][j-1], dtw[i-1][j-1]))
		}
	}
	return dtw[n][m]
}
func findAlignmentPath(a, b []Histogram, window int) [][]int {
	n, m := len(a), len(b)
	dtw := make([][]float64, n+1)
	for i := range dtw {
		dtw[i] = make([]float64, m+1)
		for j := range dtw[i] {
			dtw[i][j] = math.Inf(1)
		}
	}
	dtw[0][0] = 0

	for i := 1; i <= n; i++ {
		for j := max(1, i-window); j <= min(m, i+window); j++ {
			cost := calculateDistance(a[i-1], b[j-1])
			dtw[i][j] = cost + math.Min(dtw[i-1][j], math.Min(dtw[i][j-1], dtw[i-1][j-1]))
		}
	}

	path := [][]int{}
	i, j := n, m
	for i > 0 && j > 0 {
		path = append([][]int{{i - 1, j - 1}}, path...)
		if dtw[i-1][j] < dtw[i][j-1] && dtw[i-1][j] < dtw[i-1][j-1] {
			i--
		} else if dtw[i][j-1] < dtw[i-1][j] && dtw[i][j-1] < dtw[i-1][j-1] {
			j--
		} else {
			i--
			j--
		}
	}
	return path
}

func DTW(AQ, BQ []Histogram) ([][]int, float64) {
	n := len(AQ)
	m := len(BQ)
	dtw := make([][]float64, n+1)
	for i := range dtw {
		dtw[i] = make([]float64, m+1)
		for j := range dtw[i] {
			dtw[i][j] = math.Inf(1)
		}
	}
	dtw[0][0] = 0

	for i := 1; i <= n; i++ {
		for j := 1; j <= m; j++ {
			cost := AQ[i-1].dotProduct(BQ[j-1])
			dtw[i][j] = cost + math.Min(dtw[i-1][j], math.Min(dtw[i][j-1], dtw[i-1][j-1]))
		}
	}

	path := [][]int{}
	i, j := n, m
	for i > 0 && j > 0 {
		path = append([][]int{{i - 1, j - 1}}, path...)
		if dtw[i-1][j] < dtw[i][j-1] && dtw[i-1][j] < dtw[i-1][j-1] {
			i--
		} else if dtw[i][j-1] < dtw[i-1][j] && dtw[i][j-1] < dtw[i-1][j-1] {
			j--
		} else {
			i--
			j--
		}
	}

	return path, dtw[n][m]
}

func cipherFrame(fileA, fileB, targetDir string, path [][]int) {
	capA, err := gocv.VideoCaptureFile(fileA)
	if err != nil {
		fmt.Println("Error opening video file: ", fileA)
		return
	}
	defer capA.Close()

	capB, err := gocv.VideoCaptureFile(fileB)
	if err != nil {
		fmt.Println("Error opening video file:", fileB)
		return
	}
	defer capB.Close()

	// 获取视频属性
	width := int(capA.Get(gocv.VideoCaptureFrameWidth))
	height := int(capA.Get(gocv.VideoCaptureFrameHeight))
	fps := capA.Get(gocv.VideoCaptureFPS)

	// 创建视频写入器
	writerA, err := gocv.VideoWriterFile(targetDir+fileA+"_dwt_align.mp4", "mp4v", fps, width, height, true)
	if err != nil {
		fmt.Println("Error opening video writer: A_align.mp4")
		return
	}
	defer writerA.Close()

	writerB, err := gocv.VideoWriterFile(targetDir+fileB+"_dwt_align.mp4", "mp4v", fps, width, height, true)
	if err != nil {
		fmt.Println("Error opening video writer: B_align.mp4")
		return
	}
	defer writerB.Close()

	// 读取视频帧并存储在切片中
	var framesA []gocv.Mat
	for {
		frame := gocv.NewMat()
		if !capA.Read(&frame) {
			break
		}
		framesA = append(framesA, frame)
	}
	var framesB []gocv.Mat
	for {
		frame := gocv.NewMat()
		if !capB.Read(&frame) {
			break
		}
		framesB = append(framesB, frame)
	}

	// 根据 DTW 路径同步视频帧
	for _, p := range path {
		frameA := framesA[p[0]]
		frameB := framesB[p[1]]
		writerA.Write(frameA)
		writerB.Write(frameB)
	}

	fmt.Println("Video alignment completed.")
}
func interpolateFrames(frame1, frame2 gocv.Mat, alpha float64) gocv.Mat {
	result := gocv.NewMat()
	gocv.AddWeighted(frame1, alpha, frame2, 1-alpha, 0, &result)
	return result
}

func cipherFrame2(fileA, fileB, targetDir string, path [][]int) {
	capA, err := gocv.VideoCaptureFile(fileA)
	if err != nil {
		fmt.Println("Error opening video file: ", fileA)
		return
	}
	defer capA.Close()

	capB, err := gocv.VideoCaptureFile(fileB)
	if err != nil {
		fmt.Println("Error opening video file:", fileB)
		return
	}
	defer capB.Close()

	// 获取视频属性
	width := int(capA.Get(gocv.VideoCaptureFrameWidth))
	height := int(capA.Get(gocv.VideoCaptureFrameHeight))
	fps := capA.Get(gocv.VideoCaptureFPS)

	// 创建视频写入器
	writerA, err := gocv.VideoWriterFile(targetDir+fileA+"_align.mp4", "mp4v", fps, width, height, true)
	if err != nil {
		fmt.Println("Error opening video writer: A_align.mp4")
		return
	}
	defer writerA.Close()

	writerB, err := gocv.VideoWriterFile(targetDir+fileB+"_align.mp4", "mp4v", fps, width, height, true)
	if err != nil {
		fmt.Println("Error opening video writer: B_align.mp4")
		return
	}
	defer writerB.Close()

	// 读取视频帧并存储在切片中
	var framesA []gocv.Mat
	for {
		frame := gocv.NewMat()
		if !capA.Read(&frame) {
			break
		}
		framesA = append(framesA, frame)
	}
	var framesB []gocv.Mat
	for {
		frame := gocv.NewMat()
		if !capB.Read(&frame) {
			break
		}
		framesB = append(framesB, frame)
	}

	// 根据 DTW 路径同步视频帧
	var prevA, prevB gocv.Mat
	for k, p := range path {
		frameA := framesA[p[0]]
		frameB := framesB[p[1]]
		if k > 0 && p[0] == path[k-1][0] {
			// A视频帧重复，插值生成新帧
			frameA = interpolateFrames(prevA, frameA, 0.5)
		}
		if k > 0 && p[1] == path[k-1][1] {
			// B视频帧重复，插值生成新帧
			frameB = interpolateFrames(prevB, frameB, 0.5)
		}
		writerA.Write(frameA)
		writerB.Write(frameB)
		prevA, prevB = frameA, frameB
	}
	fmt.Println("Video alignment completed.")
}

func GradientOfBlockInOneFrame(S_0 int, file string) {
	cellSize := S_0 / Cell_M
	blockSize := cellSize / Cell_m
	read2FrameFromSameVideo(file, func(w, h float64, a, b, x, y, t *gocv.Mat) {
		//_ = calculateCenters(w, h, float64(S_0))
		width, height := int(w), int(h)
		var blockWidth = (width + blockSize - 1) / blockSize
		var blockHeight = (height + blockSize - 1) / blockSize

		var blockGradient = make([][]Histogram, blockHeight)

		for rowIdx := 0; rowIdx < blockHeight; rowIdx++ {
			blockGradient[rowIdx] = make([]Histogram, blockWidth)
			for colIdx := 0; colIdx < blockWidth; colIdx++ {
				blockGradient[rowIdx][colIdx] = quantizeGradientOfBlock(rowIdx, colIdx, blockSize, width, height, x, y, t)
			}
		}

		saveJson(fmt.Sprintf("tmp/ios/cpu_block_gradient_one_frame_%s_%d.json", file, S_0), blockGradient)
	})
}

func DescriptorOfOneCenter(gradient [][]Histogram, size int, weights [][]float64) {

	blockNumInRoi := Cell_M * Cell_m
	for k := 10; k < 12; k++ {

		var weightedBlockGradientInDesc = make([][]Histogram, blockNumInRoi)
		var descHistogram = make([]Histogram, Cell_M*Cell_M)

		for i := 0; i < blockNumInRoi; i++ {
			weightedBlockGradientInDesc[i] = make([]Histogram, blockNumInRoi)
		}

		for i := 0; i < blockNumInRoi; i++ {
			for j := 0; j < blockNumInRoi; j++ {
				whg := gradient[i+k][j+k].Scale(weights[i][j])
				weightedBlockGradientInDesc[i][j] = whg
				cellIdx := (i/Cell_m)*Cell_M + j/Cell_m
				descHistogram[cellIdx].Add(whg)
				//fmt.Println(weightedDistances[i][j], gradient[i][j], whg, descHistogram[cellIdx], "(", i, j, "),(", cellIdx, ")")
			}
		}
		saveJson(fmt.Sprintf("tmp/ios/cpu_weighted_gradient(%d)_one_descriptor_%d.json", size, k), weightedBlockGradientInDesc)
		saveJson(fmt.Sprintf("tmp/ios/cpu_gradient_one_descriptor(%d)_%d.json", size, k), descHistogram)
		normalizedDescHistogram(descHistogram)
		saveJson(fmt.Sprintf("tmp/ios/cpu_normalized_gradient_one_descriptor(%d)_%d.json", size, k), descHistogram)
	}
}

func normalizedDescHistogram(descHistogram []Histogram) {
	l2Norm := 0.0
	for _, histogram := range descHistogram {
		for _, value := range histogram {
			l2Norm += value * value
		}
	}
	l2Norm = math.Sqrt(l2Norm)

	// 正则化L2范数
	l2NormRegularized := l2Norm + 1

	// 归一化descHistogram
	for i := range descHistogram {
		for j := range descHistogram[i] {
			descHistogram[i][j] = descHistogram[i][j] / l2NormRegularized
		}
	}
}

func DescOfOneFrame(gradientA, gradientB [][]Histogram, size int, weights [][]float64) {

	height := len(gradientA)
	width := len(gradientA[0])
	blockNumOfOneRoi := Cell_M * Cell_m
	roiNumX := width - blockNumOfOneRoi + 1
	roiNumY := height - blockNumOfOneRoi + 1
	//roiGap := SideSize / Cell_M / Cell_m
	frameGradientA := make([][][Cell_M * Cell_M]Histogram, roiNumY)
	frameGradientB := make([][][Cell_M * Cell_M]Histogram, roiNumY)
	for i := 0; i < roiNumY; i++ {
		frameGradientA[i] = make([][Cell_M * Cell_M]Histogram, roiNumX)
		frameGradientB[i] = make([][Cell_M * Cell_M]Histogram, roiNumX)
		for j := 0; j < roiNumX; j++ {
			frameGradientA[i][j] = histogramOfOneDesc(i, j, width, height, gradientA, weights)
			frameGradientB[i][j] = histogramOfOneDesc(i, j, width, height, gradientB, weights)
		}
	}

	saveJson(fmt.Sprintf("tmp/ios/cpu_one_frame_all_descriptors_level_a_%d.json", size), frameGradientA)
	saveJson(fmt.Sprintf("tmp/ios/cpu_one_frame_all_descriptors_level_b_%d.json", size), frameGradientB)
	fmt.Println("desc size x:", roiNumX, width, "desc size y:", roiNumY, height)
}

func histogramOfOneDesc(rowOffset, colOffset, width, height int, gradient [][]Histogram, weights [][]float64) (oneDescHistogram [Cell_M * Cell_M]Histogram) {
	blockNumInRoi := Cell_M * Cell_m

	for i := 0; i < blockNumInRoi; i++ {
		for j := 0; j < blockNumInRoi; j++ {
			gradientRowIdx, gradientColIdx := i+rowOffset, j+colOffset
			if gradientRowIdx >= height || gradientColIdx >= width {
				fmt.Println("overflow:", i, j, rowOffset, colOffset, gradientRowIdx, gradientColIdx, width, height)
				continue
			}
			whg := gradient[gradientRowIdx][gradientColIdx].Scale(weights[i][j])
			cellIdx := (i/Cell_m)*Cell_M + j/Cell_m
			oneDescHistogram[cellIdx].Add(whg)
		}
	}

	normalizedDescHistogram(oneDescHistogram[:])
	return
}

func IosOldRoiHistogram() {
	video, _ := gocv.VideoCaptureFile("tmp/ios/align_a.mp4")
	var frameA = gocv.NewMat()
	if ok := video.Read(&frameA); !ok || frameA.Empty() {
		panic("read error")
	}

	var grayFrame = gocv.NewMat()
	gocv.CvtColor(frameA, &grayFrame, gocv.ColorRGBToGray)
	defer frameA.Close()
	roiSize := 32
	center := Point{
		X: 16,
		Y: 16,
	}
	roiA := roiGradient(grayFrame, center, roiSize)
	saveJson("tmp/ios/cpu_desc_one_roi.json", roiA)
}

func WtlOfOneCenter(blockGradientA, blockGradientB [][]Histogram, k, s int, weights [][]float64) {

	blockNumInRoi := Cell_M * Cell_m

	var descHistogramA = make([]Histogram, Cell_M*Cell_M)
	var descHistogramB = make([]Histogram, Cell_M*Cell_M)

	for i := 0; i < blockNumInRoi; i++ {
		for j := 0; j < blockNumInRoi; j++ {
			weight := weights[i][j]
			whgA := blockGradientA[i+k][j+k].Scale(weight)
			whgB := blockGradientB[i+k][j+k].Scale(weight)
			cellIdx := (i/Cell_m)*Cell_M + j/Cell_m
			descHistogramA[cellIdx].Add(whgA)
			descHistogramB[cellIdx].Add(whgB)
		}
	}

	normalizedDescHistogram(descHistogramA)
	normalizedDescHistogram(descHistogramB)
	wtlOfOneCenter := calculateWtlOfOneRoi(descHistogramA, descHistogramB)

	saveJson(fmt.Sprintf("tmp/ios/cpu_one_descriptor(%d)_to_wtl_at_%d_a.json", s, k), descHistogramA)
	saveJson(fmt.Sprintf("tmp/ios/cpu_one_descriptor(%d)_to_wtl_at_%d_b.json", s, k), descHistogramB)
	fmt.Println("wtl of one descriptor:", wtlOfOneCenter)
}

func WtlOfOneFrame(blockGradientA, blockGradientB [][]Histogram, s int, weights [][]float64) [][]float64 {
	height := len(blockGradientA)
	width := len(blockGradientA[0])
	fmt.Println("width:", width, "height", height)

	blockNumOfOneRoi := Cell_M * Cell_m
	roiNumX := width - blockNumOfOneRoi + 1
	roiNumY := height - blockNumOfOneRoi + 1
	//roiGap := SideSize / Cell_M / Cell_m
	fmt.Println("desc size x:", roiNumX, "desc size y:", roiNumY)
	descriptorA := make([][][Cell_M * Cell_M]Histogram, roiNumY)
	descriptorB := make([][][Cell_M * Cell_M]Histogram, roiNumY)
	wtlOfOneFrame := make([][]float64, roiNumY)
	for i := 0; i < roiNumY; i++ {
		wtlOfOneFrame[i] = make([]float64, roiNumX)
		descriptorB[i] = make([][Cell_M * Cell_M]Histogram, roiNumX)
		descriptorA[i] = make([][Cell_M * Cell_M]Histogram, roiNumX)
		for j := 0; j < roiNumX; j++ {
			descriptorA[i][j], descriptorB[i][j] = weightedBlockGradient(i, j, width, height, blockGradientA, blockGradientB, weights)
			wtlOfOneFrame[i][j] = calculateWtlOfOneRoi(descriptorA[i][j][:], descriptorB[i][j][:])
		}
	}

	saveJson(fmt.Sprintf("tmp/ios/cpu_one_frame_descriptor_level_%d_a.json", s), descriptorA)
	saveJson(fmt.Sprintf("tmp/ios/cpu_one_frame_descriptor_level_%d_b_.json", s), descriptorB)
	saveJson(fmt.Sprintf("tmp/ios/cpu_one_frame_wtl_level_%d.json", s), wtlOfOneFrame)
	__saveNormalizedData(normalizeImage(wtlOfOneFrame), fmt.Sprintf("tmp/ios/cpu_one_frame_wtl_level_%d.json.png", s))

	return wtlOfOneFrame
}

func calculateWtlOfOneRoi(histA, histB []Histogram) float64 {
	sum := 0.0

	for i := 0; i < len(histA); i++ {
		for j := 0; j < HistogramSize; j++ {
			diff := histA[i][j] - histB[i][j]
			sum += diff * diff
		}
	}

	return math.Sqrt(sum)
}

func weightedBlockGradient(rowOffset, colOffset, width, height int, gradientA, gradientB [][]Histogram, weights [][]float64) (descriptorA, descriptorB [Cell_M * Cell_M]Histogram) {
	blockNumInRoi := Cell_M * Cell_m

	for i := 0; i < blockNumInRoi; i++ {
		for j := 0; j < blockNumInRoi; j++ {
			gradientRowIdx, gradientColIdx := i+rowOffset, j+colOffset
			if gradientRowIdx >= height || gradientColIdx >= width {
				fmt.Println("overflow:", i, j, gradientRowIdx, gradientColIdx, width, height)
				continue
			}
			weight := weights[i][j]
			weightedHistogramA := gradientA[gradientRowIdx][gradientColIdx].Scale(weight)
			weightedHistogramB := gradientB[gradientRowIdx][gradientColIdx].Scale(weight)
			cellIdx := (i/Cell_m)*Cell_M + j/Cell_m
			descriptorA[cellIdx].Add(weightedHistogramA)
			descriptorB[cellIdx].Add(weightedHistogramB)
		}
	}

	normalizedDescHistogram(descriptorA[:])
	normalizedDescHistogram(descriptorB[:])
	return
}

func BiLinearInterpolate(width, height int) {
	var wtl [][]float64
	var wtl2 [][]float64
	var wtl4 [][]float64
	readJson("tmp/ios/cpu_one_frame_wtl_level_32.json", &wtl)
	readJson("tmp/ios/cpu_one_frame_wtl_level_64.json", &wtl2)
	readJson("tmp/ios/cpu_one_frame_wtl_level_128.json", &wtl4)

	result := make([][]float64, height)
	result2 := make([][]float64, height)
	result4 := make([][]float64, height)
	resultCombined := make([][]float64, height)

	for y := 0; y < height; y++ {
		result[y] = make([]float64, width)
		result2[y] = make([]float64, width)
		result4[y] = make([]float64, width)
		resultCombined[y] = make([]float64, width)
		for x := 0; x < width; x++ {
			result[y][x] = bilinearInterpolate2(float64(x)/float64(32), float64(y)/float64(32), wtl, width, height)
			result2[y][x] = bilinearInterpolate2(float64(x)/float64(64), float64(y)/float64(64), wtl2, width, height)
			result4[y][x] = bilinearInterpolate2(float64(x)/float64(128), float64(y)/float64(128), wtl4, width, height)
			resultCombined[y][x] = result[y][x] + 2*result2[y][x] + 4*result4[y][x]
		}
	}

	result = normalizeImage(result)
	result2 = normalizeImage(result2)
	result4 = normalizeImage(result4)
	resultCombined = normalizeImage(resultCombined)

	__saveNormalizedData(result, fmt.Sprintf("tmp/ios/cpu_one_frame_bilinear_level_%d.json.png", 32))
	__saveNormalizedData(result2, fmt.Sprintf("tmp/ios/cpu_one_frame_bilinear_level_%d.json.png", 64))
	__saveNormalizedData(result4, fmt.Sprintf("tmp/ios/cpu_one_frame_bilinear_level_%d.json.png", 128))
	__saveNormalizedData(resultCombined, fmt.Sprintf("tmp/ios/cpu_one_frame_bilinear_combined.json.png"))

	saveJson(fmt.Sprintf("tmp/ios/cpu_one_frame_bilinear_level_%d.json", 32), result)
	saveJson(fmt.Sprintf("tmp/ios/cpu_one_frame_bilinear_level_%d.json", 64), result2)
	saveJson(fmt.Sprintf("tmp/ios/cpu_one_frame_bilinear_level_%d.json", 128), result4)
	saveJson("tmp/ios/cpu_one_frame_bilinear_combined.json", resultCombined)
}

func OverlayForOneFrame() {
	var wtl [][]float64
	readJson("tmp/ios/cpu_one_frame_bilinear_combined.json", &wtl)

	vA, vB, _ := readFile(param.alignedAFile, param.alignedBFile)
	var frameA = gocv.NewMat()
	var frameB = gocv.NewMat()
	vA.Read(&frameA)
	vB.Read(&frameB)

	var grayFrameA = gocv.NewMat()
	var grayFrameB = gocv.NewMat()
	gocv.CvtColor(frameA, &grayFrameA, gocv.ColorRGBToGray)
	gocv.CvtColor(frameB, &grayFrameB, gocv.ColorRGBToGray)
	frameA.Close()
	frameB.Close()
	gradientMagnitude := computeG(grayFrameB)
	img := overlay(grayFrameA, wtl, gradientMagnitude)
	file, _ := os.Create(fmt.Sprintf("tmp/ios/cpu_one_frame_overlay.png"))
	_ = png.Encode(file, img)
	_ = file.Close()
}

func OverlayOneFrameFromStart() {
	videoA, videoB, err := readFile(param.alignedAFile, param.alignedBFile)
	if err != nil {
		panic(err)
	}
	var descriptorSideSizeAtZeroLevel = 32
	width := int(videoA.Get(gocv.VideoCaptureFrameWidth))
	height := int(videoA.Get(gocv.VideoCaptureFrameHeight))
	var counter = 0
	//for {
	//	counter++
	frameA, frameB := gocv.NewMat(), gocv.NewMat()
	ok := videoA.Read(&frameA)
	if !ok || frameA.Empty() {
		return
	}
	videoB.Read(&frameB)
	grayFrameA, grayFrameB := gocv.NewMat(), gocv.NewMat()
	grayFrameAPre, grayFrameBPre := gocv.NewMat(), gocv.NewMat()

	gocv.CvtColor(frameA, &grayFrameAPre, gocv.ColorRGBToGray)
	gocv.CvtColor(frameB, &grayFrameBPre, gocv.ColorRGBToGray)

	//__saveImg(grayFrameAPre, "tmp/ios/overlays/cpu_one_frame_overlay_at_once_a.png")
	//__saveImg(grayFrameBPre, "tmp/ios/overlays/cpu_one_frame_overlay_at_once_b.png")

	videoA.Read(&frameA)
	videoB.Read(&frameB)

	gocv.CvtColor(frameA, &grayFrameA, gocv.ColorRGBToGray)
	gocv.CvtColor(frameB, &grayFrameB, gocv.ColorRGBToGray)

	blockGradientA := avgBlockGradientFromVideo(width, height, grayFrameAPre, grayFrameA, descriptorSideSizeAtZeroLevel)
	blockGradientB := avgBlockGradientFromVideo(width, height, grayFrameBPre, grayFrameB, descriptorSideSizeAtZeroLevel)
	var wtls [3][][]float64
	for i := 0; i < 3; i++ {
		wtls[i] = wtlAtOneLevel(blockGradientA[i], blockGradientB[i], weightsWithDistance[i])
	}

	wwtl := combinedPixelWt(width, height, descriptorSideSizeAtZeroLevel, wtls)
	gradientMagnitude := computeG(grayFrameB)
	img := overlay(grayFrameA, wwtl, gradientMagnitude)
	file, _ := os.Create(fmt.Sprintf("tmp/ios/overlays/cpu_one_frame_overlay_at_once_%d.png", counter))
	_ = png.Encode(file, img)
	_ = file.Close()
	//}
}

func avgBlockGradientFromVideo(width, height int, frameGrayPre, grayFrame gocv.Mat, descriptorSizeLevel0 int) (blockGradients [3][][]Histogram) {

	gradX := gocv.NewMat()
	gradY := gocv.NewMat()
	gradT := gocv.NewMat()

	gocv.Sobel(grayFrame, &gradX, gocv.MatTypeCV16S, 1, 0, 3, 1, 0, gocv.BorderDefault)
	gocv.Sobel(grayFrame, &gradY, gocv.MatTypeCV16S, 0, 1, 3, 1, 0, gocv.BorderDefault)
	gocv.AbsDiff(frameGrayPre, grayFrame, &gradT)

	for i := 0; i < 3; i++ {
		blockSize := (descriptorSizeLevel0 << i) / Cell_M / Cell_m
		var numberOfX = (width + blockSize - 1) / blockSize
		var numberOfY = (height + blockSize - 1) / blockSize
		var block = make([][]Histogram, numberOfY)
		for rowIdx := 0; rowIdx < numberOfY; rowIdx++ {
			block[rowIdx] = make([]Histogram, numberOfX)
			for colIdx := 0; colIdx < numberOfX; colIdx++ {
				hg := quantizeGradientOfBlock(rowIdx, colIdx, blockSize, width, height, &gradX, &gradY, &gradT)
				block[rowIdx][colIdx] = hg
			}
		}
		blockGradients[i] = block
	}

	gradX.Close()
	gradY.Close()
	gradT.Close()

	return
}

func wtlAtOneLevel(blockGradientA, blockGradientB [][]Histogram, weights [][]float64) [][]float64 {
	height := len(blockGradientA)
	width := len(blockGradientA[0])

	blockNumOfOneRoi := Cell_M * Cell_m
	roiNumX := width - blockNumOfOneRoi + 1
	roiNumY := height - blockNumOfOneRoi + 1
	descriptorA := make([][][Cell_M * Cell_M]Histogram, roiNumY)
	descriptorB := make([][][Cell_M * Cell_M]Histogram, roiNumY)
	wtlOfOneFrame := make([][]float64, roiNumY)
	for i := 0; i < roiNumY; i++ {
		wtlOfOneFrame[i] = make([]float64, roiNumX)
		descriptorB[i] = make([][Cell_M * Cell_M]Histogram, roiNumX)
		descriptorA[i] = make([][Cell_M * Cell_M]Histogram, roiNumX)
		for j := 0; j < roiNumX; j++ {
			descriptorA[i][j], descriptorB[i][j] = weightedBlockGradient(i, j, width, height, blockGradientA, blockGradientB, weights)
			wtlOfOneFrame[i][j] = calculateWtlOfOneRoi(descriptorA[i][j][:], descriptorB[i][j][:])
		}
	}

	return normalizeImage(wtlOfOneFrame)
}

func combinedPixelWt(width, height, descriptorSide int, wtls [3][][]float64) [][]float64 {
	resultCombined := make([][]float64, height)
	sideZero := descriptorSide
	sideOne := descriptorSide << 1
	sideTwo := descriptorSide << 2
	for y := 0; y < height; y++ {
		resultCombined[y] = make([]float64, width)
		for x := 0; x < width; x++ {
			pixelAtZero := bilinearInterpolate2(float64(x)/float64(sideZero), float64(y)/float64(sideZero), wtls[0], width, height)
			pixelAtOne := bilinearInterpolate2(float64(x)/float64(sideOne), float64(y)/float64(sideOne), wtls[1], width, height)
			pixelAtTwo := bilinearInterpolate2(float64(x)/float64(sideTwo), float64(y)/float64(sideTwo), wtls[2], width, height)
			resultCombined[y][x] = pixelAtZero + 2*pixelAtOne + 4*pixelAtTwo
		}
	}

	return normalizeImage(resultCombined)
}

func CompareVideoFromStart() {
	now := time.Now()
	videoA, videoB, err := readFile(param.alignedAFile, param.alignedBFile)
	if err != nil {
		panic(err)
	}
	var descriptorSideSizeAtZeroLevel = 32
	width := int(videoA.Get(gocv.VideoCaptureFrameWidth))
	height := int(videoA.Get(gocv.VideoCaptureFrameHeight))
	fps := videoA.Get(gocv.VideoCaptureFPS)
	fmt.Println("fps:", fps, param.alignedAFile, param.alignedBFile)
	videoWriter, _ := gocv.VideoWriterFile(
		"tmp/ios/overlay_result.mp4", // 输出视频文件
		"mp4v",                       // 编码格式
		fps,                          // FPS
		int(width),                   // 视频宽度
		int(height),                  // 视频高度
		true)                         // 是否彩色
	defer videoWriter.Close()
	var counter = 0
	var preFrameA *gocv.Mat = nil
	var preFrameB *gocv.Mat = nil
	for {
		var frameA = gocv.NewMat()
		var frameB = gocv.NewMat()
		if ok := videoA.Read(&frameA); !ok || frameA.Empty() {
			frameA.Close()
			break
		}
		if ok := videoB.Read(&frameB); !ok || frameB.Empty() {
			videoB.Close()
			break
		}

		grayFrameA, grayFrameB := gocv.NewMat(), gocv.NewMat()
		gocv.CvtColor(frameA, &grayFrameA, gocv.ColorRGBToGray)
		gocv.CvtColor(frameB, &grayFrameB, gocv.ColorRGBToGray)
		frameA.Close()
		frameB.Close()

		if preFrameA == nil || preFrameB == nil {
			preFrameA = &grayFrameA
			preFrameB = &grayFrameB
			fmt.Println("start get first frame")
			continue
		}

		blockGradientA := avgBlockGradientFromVideo(width, height, *preFrameA, grayFrameA, descriptorSideSizeAtZeroLevel)
		blockGradientB := avgBlockGradientFromVideo(width, height, *preFrameB, grayFrameB, descriptorSideSizeAtZeroLevel)

		var wtls [3][][]float64
		for i := 0; i < 3; i++ {
			wtls[i] = wtlAtOneLevel(blockGradientA[i], blockGradientB[i], weightsWithDistance[i])
		}

		wwtl := combinedPixelWt(width, height, descriptorSideSizeAtZeroLevel, wtls)
		gradientMagnitude := computeG(grayFrameB)
		img := overlay(grayFrameA, wwtl, gradientMagnitude)
		mat, err := gocv.ImageToMatRGB(img)
		if err != nil {
			panic(err)
		}
		_ = videoWriter.Write(mat)

		mat.Close()
		preFrameA.Close()
		preFrameB.Close()
		preFrameA = &grayFrameA
		preFrameB = &grayFrameB
		counter++
		fmt.Println("finish frame:", counter)
	}
	fmt.Println("time used:", time.Now().Sub(now))
}

func FilteredVideoDiff() {
	now := time.Now()
	videoA, videoB, err := readFile(param.alignedAFile, param.alignedBFile)
	if err != nil {
		panic(err)
	}
	var descriptorSideSizeAtZeroLevel = 32
	width := int(videoA.Get(gocv.VideoCaptureFrameWidth))
	height := int(videoA.Get(gocv.VideoCaptureFrameHeight))
	fps := videoA.Get(gocv.VideoCaptureFPS)
	fmt.Println("fps:", fps, param.alignedAFile, param.alignedBFile)
	videoWriter, _ := gocv.VideoWriterFile(
		"tmp/ios/overlay_result.mp4", // 输出视频文件
		"mp4v",                       // 编码格式
		fps,                          // FPS
		int(width),                   // 视频宽度
		int(height),                  // 视频高度
		true)                         // 是否彩色
	defer videoWriter.Close()
	var counter = 0
	var preFrameA *gocv.Mat = nil
	var preFrameB *gocv.Mat = nil
	var preWwtl [][]float64 = nil

	for {
		var frameA = gocv.NewMat()
		var frameB = gocv.NewMat()
		if ok := videoA.Read(&frameA); !ok || frameA.Empty() {
			frameA.Close()
			break
		}
		if ok := videoB.Read(&frameB); !ok || frameB.Empty() {
			videoB.Close()
			break
		}

		grayFrameA, grayFrameB := gocv.NewMat(), gocv.NewMat()
		gocv.CvtColor(frameA, &grayFrameA, gocv.ColorRGBToGray)
		gocv.CvtColor(frameB, &grayFrameB, gocv.ColorRGBToGray)
		frameA.Close()
		frameB.Close()

		if preFrameA == nil || preFrameB == nil {
			preFrameA = &grayFrameA
			preFrameB = &grayFrameB
			fmt.Println("start get first frame")
			continue
		}

		blockGradientA := avgBlockGradientFromVideo(width, height, *preFrameA, grayFrameA, descriptorSideSizeAtZeroLevel)
		blockGradientB := avgBlockGradientFromVideo(width, height, *preFrameB, grayFrameB, descriptorSideSizeAtZeroLevel)

		var wtls [3][][]float64
		for i := 0; i < 3; i++ {
			wtls[i] = wtlAtOneLevel(blockGradientA[i], blockGradientB[i], weightsWithDistance[i])
		}

		wwtl := combinedPixelWt(width, height, descriptorSideSizeAtZeroLevel, wtls)

		// 简单的高通滤波：当前帧减去前一帧的权重
		if preWwtl != nil {
			for i := range wwtl {
				for j := range wwtl[i] {
					wwtl[i][j] = math.Abs(wwtl[i][j] - preWwtl[i][j]) // 使用绝对值
				}
			}
		}

		gradientMagnitude := computeG(grayFrameB)
		img := overlay(grayFrameA, wwtl, gradientMagnitude)
		mat, err := gocv.ImageToMatRGB(img)
		if err != nil {
			panic(err)
		}
		_ = videoWriter.Write(mat)

		mat.Close()
		preFrameA.Close()
		preFrameB.Close()
		preFrameA = &grayFrameA
		preFrameB = &grayFrameB
		counter++
		preWwtl = wwtl

		fmt.Println("finish frame:", counter)
	}
	fmt.Println("time used:", time.Now().Sub(now))
}

func FilteredVideoDiff2() {
	now := time.Now()
	videoA, videoB, err := readFile(param.alignedAFile, param.alignedBFile)
	if err != nil {
		panic(err)
	}
	var descriptorSideSizeAtZeroLevel = 32
	width := int(videoA.Get(gocv.VideoCaptureFrameWidth))
	height := int(videoA.Get(gocv.VideoCaptureFrameHeight))
	fps := videoA.Get(gocv.VideoCaptureFPS)
	fmt.Println("fps:", fps, param.alignedAFile, param.alignedBFile)
	videoWriter, _ := gocv.VideoWriterFile(
		"tmp/ios/overlay_result.mp4", // 输出视频文件
		"mp4v",                       // 编码格式
		fps,                          // FPS
		int(width),                   // 视频宽度
		int(height),                  // 视频高度
		true)                         // 是否彩色
	defer videoWriter.Close()
	var counter = 0
	var preFrameA *gocv.Mat = nil
	var preFrameB *gocv.Mat = nil
	movingAverage := NewMovingAverage(5) // 移动平均窗口大小为5

	for {
		var frameA = gocv.NewMat()
		var frameB = gocv.NewMat()
		if ok := videoA.Read(&frameA); !ok || frameA.Empty() {
			frameA.Close()
			break
		}
		if ok := videoB.Read(&frameB); !ok || frameB.Empty() {
			videoB.Close()
			break
		}

		grayFrameA, grayFrameB := gocv.NewMat(), gocv.NewMat()
		gocv.CvtColor(frameA, &grayFrameA, gocv.ColorRGBToGray)
		gocv.CvtColor(frameB, &grayFrameB, gocv.ColorRGBToGray)
		frameA.Close()
		frameB.Close()

		if preFrameA == nil || preFrameB == nil {
			preFrameA = &grayFrameA
			preFrameB = &grayFrameB
			fmt.Println("start get first frame")
			continue
		}

		blockGradientA := avgBlockGradientFromVideo(width, height, *preFrameA, grayFrameA, descriptorSideSizeAtZeroLevel)
		blockGradientB := avgBlockGradientFromVideo(width, height, *preFrameB, grayFrameB, descriptorSideSizeAtZeroLevel)

		var wtls [3][][]float64
		for i := 0; i < 3; i++ {
			wtls[i] = wtlAtOneLevel(blockGradientA[i], blockGradientB[i], weightsWithDistance[i])
		}

		wwtl := combinedPixelWt(width, height, descriptorSideSizeAtZeroLevel, wtls)
		// 将当前帧的wwtl添加到移动平均滤波器
		movingAverage.AddFrame(wwtl)

		// 获取移动平均值
		averageWwtl := movingAverage.GetAverage()
		if averageWwtl != nil {
			// 计算当前帧与移动平均值的差值
			for i := range wwtl {
				for j := range wwtl[i] {
					wwtl[i][j] = math.Abs(wwtl[i][j] - averageWwtl[i][j])
				}
			}
		}

		gradientMagnitude := computeG(grayFrameB)
		img := overlay(grayFrameA, wwtl, gradientMagnitude)
		mat, err := gocv.ImageToMatRGB(img)
		if err != nil {
			panic(err)
		}
		_ = videoWriter.Write(mat)

		mat.Close()
		preFrameA.Close()
		preFrameB.Close()
		preFrameA = &grayFrameA
		preFrameB = &grayFrameB
		counter++

		fmt.Println("finish frame:", counter)
	}
	fmt.Println("time used:", time.Now().Sub(now))
}

func ComputeWeightedDescriptor(S_0 int, weights [][]float64, file string) (normalizedDescriptor [][][]float64) {
	blockSize := S_0 / Cell_M / Cell_m

	read2FrameFromSameVideo(file, func(w, h float64, a, b, x, y, t *gocv.Mat) {
		width, height := int(w), int(h)

		var colsOfBlock = (width + blockSize - 1) / blockSize
		var rowsOfBlock = (height + blockSize - 1) / blockSize

		var blockGradient = make([][]Histogram, rowsOfBlock)
		var blockNumOneDescriptor = Cell_M * Cell_m

		var descriptorRows = rowsOfBlock - blockNumOneDescriptor + 1
		var descriptorCols = colsOfBlock - blockNumOneDescriptor + 1

		descriptor := make([][][Cell_M * Cell_M]Histogram, descriptorRows)
		normalizedDescriptor = make([][][]float64, descriptorRows)
		for i := 0; i < descriptorRows; i++ {
			descriptor[i] = make([][Cell_M * Cell_M]Histogram, descriptorCols)
			normalizedDescriptor[i] = make([][]float64, descriptorCols)
		}

		for rowIdx := 0; rowIdx < rowsOfBlock; rowIdx++ {
			blockGradient[rowIdx] = make([]Histogram, colsOfBlock)
			for colIdx := 0; colIdx < colsOfBlock; colIdx++ {
				hg := quantizeGradientOfBlock(rowIdx, colIdx, blockSize, width, height, x, y, t)
				blockGradient[rowIdx][colIdx] = hg

			}
		}

		saveJson(fmt.Sprintf("tmp/ios/overlays/cpu_gradients_one_frame_level_%d.json", S_0), blockGradient)
		__histogramToImg(blockGradient, fmt.Sprintf("tmp/ios/overlays/cpu_gradients_one_frame_level_%d.json.png", S_0))

		for rowIdx := 0; rowIdx < descriptorRows; rowIdx++ {
			for colIdx := 0; colIdx < descriptorCols; colIdx++ {
				blockGradientStartRowIdx := rowIdx
				blockGradientStartColIdx := colIdx

				for wRowIdx := 0; wRowIdx < blockNumOneDescriptor; wRowIdx++ {
					for wColIdx := 0; wColIdx < blockNumOneDescriptor; wColIdx++ {
						weight := weights[wRowIdx][wColIdx]
						gradient := blockGradient[blockGradientStartRowIdx+wRowIdx][blockGradientStartColIdx+wColIdx]
						weightedGradient := gradient.Scale(weight)
						cellIdxInDescriptor := (wRowIdx/Cell_m)*Cell_M + wColIdx/Cell_m
						descriptor[rowIdx][colIdx][cellIdxInDescriptor].Add(weightedGradient)
					}
				}
			}
		}

		for rowIdx, rowData := range descriptor {
			for colIdx, datum := range rowData {
				normalizedDescriptor[rowIdx][colIdx] = normalizeDescriptor(datum)
			}
		}

		saveJson(fmt.Sprintf("tmp/ios/overlays/cpu_descriptors_one_frame_level_%d.json", S_0), descriptor)
		saveJson(fmt.Sprintf("tmp/ios/overlays/cpu_descriptors_one_frame_normalized_level_%d.json", S_0), normalizedDescriptor)
		__histogramToImgFloat(normalizedDescriptor, fmt.Sprintf("tmp/ios/overlays/cpu_descriptors_one_frame_normalized_level_%d.json.png", S_0))
	})
	return
}

// normalizeDescriptor 归一化描述符，对整个描述符进行L2归一化
func normalizeDescriptor(descriptor [Cell_M * Cell_M]Histogram) []float64 {
	descriptorLength := Cell_M * Cell_M * HistogramSize

	l2Norm := 0.0
	flatDescriptor := make([]float64, descriptorLength)
	index := 0
	for i := 0; i < Cell_M*Cell_M; i++ {
		for j := 0; j < HistogramSize; j++ {
			var value = descriptor[i][j]
			flatDescriptor[index] = value
			l2Norm += value * value
			index++
		}
	}
	l2Norm = math.Sqrt(l2Norm) + 1

	// 将每个元素除以L2范数
	for i := range flatDescriptor {
		flatDescriptor[i] /= l2Norm
	}
	return flatDescriptor
}

func BiLinearWeightedDescriptor(S_0 int, weights [][]float64) [][]float64 {

	videoA, _ := gocv.VideoCaptureFile(param.alignedAFile)

	width := int(videoA.Get(gocv.VideoCaptureFrameWidth))
	height := int(videoA.Get(gocv.VideoCaptureFrameHeight))
	descOfA := ComputeWeightedDescriptor(S_0, weights, param.alignedAFile)
	descOfB := ComputeWeightedDescriptor(S_0, weights, param.alignedBFile)

	descRowLen := len(descOfA)
	descColLen := len(descOfA[0])
	wtl := make([][]float64, descRowLen)
	for rowIdx := 0; rowIdx < descRowLen; rowIdx++ {
		wtl[rowIdx] = make([]float64, descColLen)
		for colIdx := 0; colIdx < descColLen; colIdx++ {
			wtl[rowIdx][colIdx] = calculateEuclideanDistance(descOfA[rowIdx][colIdx], descOfB[rowIdx][colIdx])
		}
	}
	saveJson(fmt.Sprintf("tmp/ios/overlays/cpu_wtl_one_frame_level_%d.json", S_0), wtl)

	//wtlFullImg := applyBiLinearInterpolationToGrid(wtl, width, height, S_0)
	wtlFullImg := applyBiLinearInterpolationToFullFrame(wtl, width, height, S_0)

	saveJson(fmt.Sprintf("tmp/ios/overlays/cpu_wtl_full_one_frame_level_%d.json", S_0), wtlFullImg)

	__saveNormalizedData(normalizeImage(wtl), fmt.Sprintf("tmp/ios/overlays/cpu_wtl_one_frame_level_%d.json.png", S_0))
	__saveNormalizedData(normalizeImage(wtlFullImg), fmt.Sprintf("tmp/ios/overlays/cpu_wtl_full_one_frame_level_%d.json.png", S_0))

	return wtlFullImg

}

func calculateEuclideanDistance(descriptorA, descriptorB []float64) float64 {
	sumSquares := 0.0
	for i := 0; i < len(descriptorA); i++ {
		diff := descriptorA[i] - descriptorB[i]
		sumSquares += diff * diff
	}
	return math.Sqrt(sumSquares)
}

func biLinearInterpolate(q11, q12, q21, q22, x1, x2, y1, y2, x, y float64) float64 {
	denom := (x2 - x1) * (y2 - y1)
	intermed := q11*(x2-x)*(y2-y) + q21*(x-x1)*(y2-y) + q12*(x2-x)*(y-y1) + q22*(x-x1)*(y-y1)
	return intermed / denom
}

func applyBiLinearInterpolationToGrid(wtl [][]float64, width, height, S_0 int) [][]float64 {
	fullMap := make([][]float64, height)
	for i := range fullMap {
		fullMap[i] = make([]float64, width)
	}
	blockSize := S_0 / Cell_M / Cell_m

	// Assuming wtl grid aligns with blockSize
	for i := 0; i < len(wtl)-1; i++ {
		for j := 0; j < len(wtl[0])-1; j++ {
			x1, x2 := j*blockSize, (j+1)*blockSize
			y1, y2 := i*blockSize, (i+1)*blockSize
			for x := x1; x < x2; x++ {
				for y := y1; y < y2; y++ {
					fullMap[y][x] = biLinearInterpolate(
						wtl[i][j], wtl[i][j+1], wtl[i+1][j], wtl[i+1][j+1],
						float64(x1), float64(x2), float64(y1), float64(y2),
						float64(x), float64(y),
					)
				}
			}
		}
	}

	return fullMap
}

func applyBiLinearInterpolationToFullFrame(wtl [][]float64, width, height, S_0 int) [][]float64 {
	fullMap := make([][]float64, height)
	for i := range fullMap {
		fullMap[i] = make([]float64, width)
	}
	blockSize := S_0 / Cell_M / Cell_m
	shift := S_0 / 2
	for i := 0; i < len(wtl)-1; i++ {
		for j := 0; j < len(wtl[0])-1; j++ {
			x1, x2 := j*blockSize+shift, (j+1)*blockSize+shift
			y1, y2 := i*blockSize+shift, (i+1)*blockSize+shift
			for x := x1; x < x2; x++ {
				for y := y1; y < y2; y++ {
					fullMap[y][x] = biLinearInterpolate(
						wtl[i][j], wtl[i][j+1], wtl[i+1][j], wtl[i+1][j+1],
						float64(x1), float64(x2), float64(y1), float64(y2),
						float64(x), float64(y),
					)
				}
			}
		}
	}

	return fullMap
}

func WtlFromBiLinearFullMap(idx int) {

	var S_0 = 32
	fullMap32 := BiLinearWeightedDescriptor(S_0, weightsWithDistance[0])
	fullMap64 := BiLinearWeightedDescriptor(S_0<<1, weightsWithDistance[1])
	fullMap128 := BiLinearWeightedDescriptor(S_0<<2, weightsWithDistance[2])

	height := len(fullMap32)
	width := len(fullMap32[0])
	finalMap := make([][]float64, height)
	for rowIdx := 0; rowIdx < height; rowIdx++ {
		finalMap[rowIdx] = make([]float64, width)
		for colIdx := 0; colIdx < width; colIdx++ {
			finalMap[rowIdx][colIdx] = fullMap32[rowIdx][colIdx] + 2*fullMap64[rowIdx][colIdx] + 4*fullMap128[rowIdx][colIdx]
		}
	}
	saveJson(fmt.Sprintf("tmp/ios/overlays/cpu_wtl_full_combined_one_frame_level_%d.json", idx), finalMap)
	var normalizedMap = normalizeImage(finalMap)
	saveJson(fmt.Sprintf("tmp/ios/overlays/cpu_wtl_full_combined_one_frame_level_normalized_%d.json", idx), normalizedMap)
	__saveNormalizedData(normalizedMap, fmt.Sprintf("tmp/ios/overlays/cpu_wtl_full_combined_one_frame_level.json_%d.png", idx))
}

func __getQG(width, height, blockSize int, grayFramePre, grayFrame *gocv.Mat) (blockGradient [][]Histogram) {

	gradX := gocv.NewMat()
	gradY := gocv.NewMat()
	gradT := gocv.NewMat()

	gocv.Sobel(*grayFrame, &gradX, gocv.MatTypeCV16S, 1, 0, 3, 1, 0, gocv.BorderDefault)
	gocv.Sobel(*grayFrame, &gradY, gocv.MatTypeCV16S, 0, 1, 3, 1, 0, gocv.BorderDefault)
	gocv.AbsDiff(*grayFrame, *grayFramePre, &gradT)

	var colsOfBlock = (width + blockSize - 1) / blockSize
	var rowsOfBlock = (height + blockSize - 1) / blockSize
	blockGradient = make([][]Histogram, rowsOfBlock)
	for rowIdx := 0; rowIdx < rowsOfBlock; rowIdx++ {
		blockGradient[rowIdx] = make([]Histogram, colsOfBlock)
		for colIdx := 0; colIdx < colsOfBlock; colIdx++ {
			hg := quantizeGradientOfBlock(rowIdx, colIdx, blockSize, width, height, &gradX, &gradY, &gradT)
			blockGradient[rowIdx][colIdx] = hg
		}
	}

	gradX.Close()
	gradY.Close()
	gradT.Close()

	return
}
func __getDes(blockGradient [][]Histogram, blockNumOneDescriptor int, weights [][]float64) (normalizedDescriptor [][][]float64) {
	var rowsOfBlock = len(blockGradient)
	var colsOfBlock = len(blockGradient[0])
	var descriptorRows = rowsOfBlock - blockNumOneDescriptor + 1
	var descriptorCols = colsOfBlock - blockNumOneDescriptor + 1

	descriptor := make([][][Cell_M * Cell_M]Histogram, descriptorRows)
	normalizedDescriptor = make([][][]float64, descriptorRows)
	for i := 0; i < descriptorRows; i++ {
		descriptor[i] = make([][Cell_M * Cell_M]Histogram, descriptorCols)
		normalizedDescriptor[i] = make([][]float64, descriptorCols)
	}

	for rowIdx := 0; rowIdx < descriptorRows; rowIdx++ {
		for colIdx := 0; colIdx < descriptorCols; colIdx++ {
			blockGradientStartRowIdx := rowIdx
			blockGradientStartColIdx := colIdx

			for wRowIdx := 0; wRowIdx < blockNumOneDescriptor; wRowIdx++ {
				for wColIdx := 0; wColIdx < blockNumOneDescriptor; wColIdx++ {
					weight := weights[wRowIdx][wColIdx]
					gradient := blockGradient[blockGradientStartRowIdx+wRowIdx][blockGradientStartColIdx+wColIdx]
					weightedGradient := gradient.Scale(weight)
					cellIdxInDescriptor := (wRowIdx/Cell_m)*Cell_M + wColIdx/Cell_m
					descriptor[rowIdx][colIdx][cellIdxInDescriptor].Add(weightedGradient)
				}
			}
		}
	}

	for rowIdx, rowData := range descriptor {
		for colIdx, datum := range rowData {
			normalizedDescriptor[rowIdx][colIdx] = normalizeDescriptor(datum)
		}
	}

	return
}

func WtlOneFrameFromStart() {
	var normalizedDescriptorA [3][][][]float64
	var normalizedDescriptorB [3][][][]float64
	width, height := 0, 0
	var S_0 = 32
	var blockNumOneDescriptor = Cell_M * Cell_m
	var gradientMagnitude [][]float64
	var grayFrameA *gocv.Mat
	read2FrameFromSameVideo(param.alignedAFile, func(w, h float64, a, b, x, y, t *gocv.Mat) {
		width, height = int(w), int(h)
		for i := 0; i < 3; i++ {
			blockSize := (S_0 << i) / blockNumOneDescriptor
			blockGradient := __getQG(width, height, blockSize, a, b)
			normalizedDescriptorA[i] = __getDes(blockGradient, blockNumOneDescriptor, weightsWithDistance[i])
		}
		grayFrameA = a
	})

	read2FrameFromSameVideo(param.alignedBFile, func(w, h float64, a, b, x, y, t *gocv.Mat) {
		width, height = int(w), int(h)
		for i := 0; i < 3; i++ {
			blockSize := (S_0 << i) / blockNumOneDescriptor
			blockGradient := __getQG(width, height, blockSize, a, b)
			normalizedDescriptorB[i] = __getDes(blockGradient, blockNumOneDescriptor, weightsWithDistance[i])
		}

		//gradientMagnitude = computeG(*b)
		gradientMagnitude = computeG2(*b)
		b.Close()
	})

	var wtls [3][][]float64
	var wtlFullImgs [3][][]float64
	for i := 0; i < 3; i++ {
		var descRowLen = len(normalizedDescriptorA[i])
		wtls[i] = make([][]float64, descRowLen)

		for rowIdx := 0; rowIdx < descRowLen; rowIdx++ {
			var descColLen = len(normalizedDescriptorA[i][rowIdx])
			wtls[i][rowIdx] = make([]float64, descColLen)
			for colIdx := 0; colIdx < descColLen; colIdx++ {
				wtls[i][rowIdx][colIdx] = calculateEuclideanDistance(normalizedDescriptorA[i][rowIdx][colIdx], normalizedDescriptorB[i][rowIdx][colIdx])
			}
		}

		saveJson(fmt.Sprintf("tmp/ios/overlays/cpu_wtl_level_%d_.json", i), wtls[i])
		__saveNormalizedData(normalizeImage(wtls[i]), fmt.Sprintf("tmp/ios/overlays/cpu_wtl_level_%d_.json.png", i))
		wtlFullImgs[i] = applyBiLinearInterpolationToFullFrame(wtls[i], width, height, S_0<<i)
	}

	finalMap := make([][]float64, height)
	for rowIdx := 0; rowIdx < height; rowIdx++ {
		finalMap[rowIdx] = make([]float64, width)
		for colIdx := 0; colIdx < width; colIdx++ {
			finalMap[rowIdx][colIdx] = wtlFullImgs[0][rowIdx][colIdx] + 2*wtlFullImgs[1][rowIdx][colIdx] + 4*wtlFullImgs[2][rowIdx][colIdx]
		}
	}

	var normalizedMap = normalizeImage(finalMap)
	saveJson(fmt.Sprintf("tmp/ios/overlays/cpu_wtl_final_.json"), normalizedMap)
	__saveNormalizedData(normalizedMap, fmt.Sprintf("tmp/ios/overlays/cpu_wtl_final_.json.png"))

	saveJson("tmp/ios/overlays/cpu_gradient_magnitude_.json", gradientMagnitude)
	__saveNormalizedData(normalizeImage(gradientMagnitude), "tmp/ios/overlays/cpu_gradient_magnitude_.json.png")

	__saveImg(*grayFrameA, "tmp/ios/overlays/cpu_gray_frameA.png")
	img := overlay2(*grayFrameA, normalizedMap, gradientMagnitude)

	file, _ := os.Create(fmt.Sprintf("tmp/ios/overlays/cpu_one_frame_overlay.png"))
	_ = png.Encode(file, img)
	_ = file.Close()
	grayFrameA.Close()
}

func computeG2(grayFrameB gocv.Mat) [][]float64 {

	// 初始化Sobel梯度矩阵
	gradX := gocv.NewMat()
	gradY := gocv.NewMat()

	gocv.Sobel(grayFrameB, &gradX, gocv.MatTypeCV16S, 1, 0, 3, 1, 0, gocv.BorderDefault)
	gocv.Sobel(grayFrameB, &gradY, gocv.MatTypeCV16S, 0, 1, 3, 1, 0, gocv.BorderDefault)

	maxGradient := math.Sqrt(255*255 + 255*255)
	gradientMagnitude := make([][]float64, grayFrameB.Rows())
	for y := 0; y < grayFrameB.Rows(); y++ {
		gradientMagnitude[y] = make([]float64, grayFrameB.Cols())
		for x := 0; x < grayFrameB.Cols(); x++ {
			gx := float64(gradX.GetShortAt(y, x))
			gy := float64(gradY.GetShortAt(y, x))
			g := math.Sqrt(gx*gx+gy*gy) / maxGradient
			g = math.Min(1, testTool.alpha*g) // Apply alpha and cap at 1
			gradientMagnitude[y][x] = g
		}
	}
	// 清理资源
	gradX.Close()
	gradY.Close()

	return gradientMagnitude
}

func overlay2(frameA gocv.Mat, wtVal, gradientMagnitude [][]float64) image.Image {
	width := frameA.Cols()
	height := frameA.Rows()
	adjustedFrame := adjustContrastAndMap(frameA, testTool.betaLow, testTool.betaHigh)
	saveJson("tmp/ios/adjustedFrame.json", adjustedFrame)
	__saveNormalizedData(adjustedFrame, "tmp/ios/adjustedFrame.json.png")
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			score := wtVal[y][x]
			heatMapColor := fc(score)
			g := gradientMagnitude[y][x]
			grayValue := adjustedFrame[y][x]
			vColor := color.RGBA{
				R: uint8((1-g)*grayValue*255 + g*float64(heatMapColor.R)),
				G: uint8((1-g)*grayValue*255 + g*float64(heatMapColor.G)),
				B: uint8((1-g)*grayValue*255 + g*float64(heatMapColor.B)),
				A: 255,
			}
			img.Set(x, y, vColor)
		}
	}
	return img
}

func adjustContrastAndMap(frame gocv.Mat, betaLow, betaHigh float64) [][]float64 {
	// 计算百分位数
	I_low, I_high := calculatePercentiles(frame, 1, 99)
	fmt.Println("------>>>low:", I_low, "hight", I_high)
	rows, cols := frame.Rows(), frame.Cols()
	mappedFrame := make([][]float64, rows)
	for y := 0; y < rows; y++ {
		mappedFrame[y] = make([]float64, cols)
		for x := 0; x < cols; x++ {
			val := float64(frame.GetUCharAt(y, x))
			// 根据百分位数调整对比度并进行映射
			if val < I_low {
				val = betaLow
			} else if val > I_high {
				val = betaHigh
			} else {
				val = betaLow + (betaHigh-betaLow)*(val-I_low)/(I_high-I_low)
			}
			// 保证结果在0-1范围内，映射到0-255后转换为float64以便后续处理
			mappedFrame[y][x] = val
		}
	}
	return mappedFrame
}

// 计算给定百分位的像素值，简化实现，实际应用需要更复杂的直方图或排序算法
func calculatePercentiles(mat gocv.Mat, lowPerc, highPerc float64) (float64, float64) {
	hist := make([]int, 256)
	rows, cols := mat.Rows(), mat.Cols()
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := mat.GetUCharAt(y, x)
			hist[v]++
		}
	}

	total := rows * cols
	lowCount := int(float64(total) * lowPerc / 100)
	highCount := int(float64(total) * highPerc / 100)
	fmt.Println("total pixel:", total, "low:", lowCount, "high:", highCount)

	lowVal, highVal := 0, 255
	cumulative := 0
	for i, count := range hist {
		cumulative += count
		if cumulative <= lowCount {
			lowVal = i
		}
		if cumulative <= highCount {
			highVal = i
		}
	}
	saveJson("tmp/ios/cpu_percentile_2_histogram_.json", hist)
	return float64(lowVal), float64(highVal)
}

func NewCompareVideo(max int) {
	now := time.Now()
	videoA, videoB, err := readFile(param.alignedAFile, param.alignedBFile)
	if err != nil {
		panic(err)
	}
	var descriptorSideSizeAtZeroLevel = 32
	width := int(videoA.Get(gocv.VideoCaptureFrameWidth))
	height := int(videoA.Get(gocv.VideoCaptureFrameHeight))
	fps := videoA.Get(gocv.VideoCaptureFPS)
	fmt.Println("fps:", fps, param.alignedAFile, param.alignedBFile)
	videoWriter, _ := gocv.VideoWriterFile(
		"tmp/ios/overlays/overlay_result.mp4", // 输出视频文件
		"mp4v",                                // 编码格式
		fps,                                   // FPS
		int(width),                            // 视频宽度
		int(height),                           // 视频高度
		true)                                  // 是否彩色
	defer videoWriter.Close()
	var counter = 0
	var preFrameA *gocv.Mat = nil
	var preFrameB *gocv.Mat = nil
	var preFinalMap [][]float64 = nil
	for {
		var frameA = gocv.NewMat()
		var frameB = gocv.NewMat()
		if ok := videoA.Read(&frameA); !ok || frameA.Empty() {
			frameA.Close()
			break
		}
		if ok := videoB.Read(&frameB); !ok || frameB.Empty() {
			videoB.Close()
			break
		}

		grayOrigFrameA, grayOrigFrameB := gocv.NewMat(), gocv.NewMat()
		gocv.CvtColor(frameA, &grayOrigFrameA, gocv.ColorRGBToGray)
		gocv.CvtColor(frameB, &grayOrigFrameB, gocv.ColorRGBToGray)

		//__saveImg(grayFrameA, fmt.Sprintf("tmp/ios/overlays/cpu_graya_org_%d.png", counter))
		//__saveImg(grayFrameB, fmt.Sprintf("tmp/ios/overlays/cpu_grayb_org_%d.png", counter))

		var grayFrameA = SegmentForeground(grayOrigFrameA)
		var grayFrameB = SegmentForeground(grayOrigFrameB)
		if max > 0 {
			__saveImg(grayFrameA, fmt.Sprintf("tmp/ios/overlays/cpu_graya_seg_%d.png", counter))
			__saveImg(grayFrameB, fmt.Sprintf("tmp/ios/overlays/cpu_grayb_seg_%d.png", counter))
		}
		frameA.Close()
		frameB.Close()
		var blockNumOneDescriptor = Cell_M * Cell_m

		if preFrameA == nil || preFrameB == nil {
			preFrameA = &grayFrameA
			preFrameB = &grayFrameB
			fmt.Println("start get first frame")
			continue
		}

		var finalMap = make([][]float64, height)
		for i := 0; i < height; i++ {
			finalMap[i] = make([]float64, width)
		}

		for level := 0; level < 3; level++ {
			sideSize := descriptorSideSizeAtZeroLevel << level
			timer := 1 << level
			blockSize := sideSize / blockNumOneDescriptor

			var blockGradientA = __getQG(width, height, blockSize, preFrameA, &grayFrameA)
			var blockGradientB = __getQG(width, height, blockSize, preFrameB, &grayFrameB)
			var normalizedDescriptorA = __getDes(blockGradientA, blockNumOneDescriptor, weightsWithDistance[level])
			var normalizedDescriptorB = __getDes(blockGradientB, blockNumOneDescriptor, weightsWithDistance[level])

			var descRowLen = len(normalizedDescriptorA)
			var wtl = make([][]float64, descRowLen)
			for rowIdx := 0; rowIdx < descRowLen; rowIdx++ {
				var descColLen = len(normalizedDescriptorA[rowIdx])
				wtl[rowIdx] = make([]float64, descColLen)
				for colIdx := 0; colIdx < descColLen; colIdx++ {
					wtl[rowIdx][colIdx] = calculateEuclideanDistance(normalizedDescriptorA[rowIdx][colIdx], normalizedDescriptorB[rowIdx][colIdx])
				}
			}

			var wtlFullImg = applyBiLinearInterpolationToFullFrame(wtl, width, height, sideSize)
			for rowIdx := 0; rowIdx < height; rowIdx++ {
				for colIdx := 0; colIdx < width; colIdx++ {
					finalMap[rowIdx][colIdx] += wtlFullImg[rowIdx][colIdx] * float64(timer)
				}
			}
		}

		if preFinalMap == nil {
			preFinalMap = finalMap
			continue
		}
		//filteredFinalMap := simulateTemporalHighPassFilter(finalMap, preFinalMap, 0.8)

		//fMat := convertFloat64ArrayToMat32F(filteredFinalMap)
		//fMat = applyBilateralFilterToW(fMat)
		//filteredFinalMap = matToFloat64Array32F(fMat)
		var normalizedMap = normalizeImage(finalMap)
		//var normalizedMap = normalizeImage(filteredFinalMap)
		var gradientMagnitude = computeG2(grayOrigFrameB)
		img := overlay2(grayOrigFrameA, normalizedMap, gradientMagnitude)
		if max > 0 {
			__saveNormalizedData(normalizedMap, fmt.Sprintf("tmp/ios/overlays/cpu_wtl_final_%d.json.png", counter))
			file, _ := os.Create(fmt.Sprintf("tmp/ios/overlays/cpu_one_frame_overlay_%d.png", counter))
			_ = png.Encode(file, img)
			_ = file.Close()
		}
		mat, err := gocv.ImageToMatRGB(img)
		if err != nil {
			panic(err)
		}
		_ = videoWriter.Write(mat)

		mat.Close()
		preFrameA.Close()
		preFrameB.Close()
		preFrameA = &grayFrameA
		preFrameB = &grayFrameB

		fmt.Println("finish frame:", counter)
		counter++

		if counter >= max && max > 0 {
			break
		}
	}
	fmt.Println("time used:", time.Now().Sub(now))
}

// SegmentForeground 使用阈值方法分割前景
func SegmentForeground(grayFrame gocv.Mat) gocv.Mat {
	// 创建一个新的Mat来存储分割结果
	foregroundMask := gocv.NewMat()
	defer foregroundMask.Close()

	// 应用阈值处理，阈值可以根据实际情况调整
	gocv.Threshold(grayFrame, &foregroundMask, 240, 255, gocv.ThresholdBinary)

	return foregroundMask.Clone()
}

func TemporalHighPassFilter(currentGrad, previousGrad gocv.Mat) gocv.Mat {
	// 创建一个新的Mat来存储结果
	result := gocv.NewMat()
	defer result.Close()

	// 计算当前帧与前一帧的差异
	gocv.AbsDiff(currentGrad, previousGrad, &result)

	// 应用阈值，去除微小的变化
	threshold := 10.0 // 根据实际情况调整此阈值
	gocv.Threshold(result, &result, float32(threshold), 255, gocv.ThresholdBinary)

	return result.Clone()
}

// simulateTemporalHighPassFilter 模拟时间高通滤波过程
func simulateTemporalHighPassFilter(currentW, previousW [][]float64, threshold float64) [][]float64 {
	height := len(currentW)
	width := len(currentW[0])
	filteredW := make([][]float64, height)

	for i := range filteredW {
		filteredW[i] = make([]float64, width)
		for j := range filteredW[i] {
			diff := currentW[i][j] - previousW[i][j]
			if abs(diff) > threshold {
				filteredW[i][j] = diff
			} else {
				filteredW[i][j] = 0
			}
		}
	}
	return filteredW
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func applyBilateralFilterToW(filteredW gocv.Mat) gocv.Mat {
	// 创建结果Mat
	bilateralResult := gocv.NewMat()
	defer bilateralResult.Close()

	// 应用双边滤波器
	// 你可能需要根据实际情况调整双边滤波器的参数
	gocv.BilateralFilter(filteredW, &bilateralResult, -1, 75.0, 75.0)

	return bilateralResult.Clone()
}

func convertFloat64ArrayToMat32F(data [][]float64) gocv.Mat {
	rows := len(data)
	cols := len(data[0])
	mat := gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV32F)

	for i := range data {
		for j := range data[i] {
			mat.SetFloatAt(i, j, float32(data[i][j]))
		}
	}

	return mat
}

// matToFloat64Array32F 从类型为 CV_32F 的 gocv.Mat 中读取数据并转换为 [][]float64
func matToFloat64Array32F(mat gocv.Mat) [][]float64 {
	// 确保 Mat 是单通道32位浮点类型
	if mat.Type() != gocv.MatTypeCV32F {
		fmt.Println("Mat must be of type CV_32F")
		return nil
	}

	// 获取 Mat 的尺寸
	rows := mat.Rows()
	cols := mat.Cols()

	// 初始化二维浮点数组
	arr := make([][]float64, rows)
	for i := range arr {
		arr[i] = make([]float64, cols)
	}

	// 将 Mat 的数据复制到浮点数组
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			arr[i][j] = float64(mat.GetFloatAt(i, j)) // 从 Mat 读取 float32 并转换为 float64
		}
	}

	return arr
}
