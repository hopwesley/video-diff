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
func gradientToImg(file string) {
	var data [][][]float64
	var imgData [][]uint8

	_ = readJson(file, &data)

	imgData = make([][]uint8, len(data))
	for row, rowData := range data {
		imgData[row] = make([]uint8, len(rowData))
		for col, columnData := range rowData {
			var sum = 0.0
			for _, value := range columnData {
				sum += value
			}
			imgData[row][col] = uint8(int(sum*10+8) % 255)
		}
	}

	saveGrayDataData(imgData, file+".png")
}

func CompareIosAndMacQG() {
	//gradientToImg("tmp/ios/simple_quantize.json", "tmp/ios/quantizeBuffer.json")
	gradientToImg("tmp/ios/block_quantize.json")
	gradientToImg("tmp/ios/blockGradientBuffer.json")
}

func AverageGradientOfBlock(S_0 int) {
	blockSize := S_0 / DescriptorParam_M / DescriptorParam_m
	read2FrameFromSameVideo(param.rawAFile, func(w, h float64, a, b, x, y, t *gocv.Mat) {
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

func quantizeGradientOfBlock(rowIdx, colIdx, blockSize, width, height int, gradientX, gradientY, gradientT *gocv.Mat) (histogram [10]float64) {
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

type CenterPoint struct {
	indexX  int
	indexY  int
	weights [][]float64
}

func calculateCenters(width, height, S float64) (centersOfDesc []CenterPoint) {
	numberXOfBlock := DescriptorParam_M * DescriptorParam_m
	numberYOfBlock := DescriptorParam_M * DescriptorParam_m
	blockSize := S / float64(numberXOfBlock)
	sigma := float64(S) / 2.0 // Standard deviation for Gaussian kernel
	centerIdxX, centerIdxY := 0, 0
	for centerY := S / 2; centerY < height; centerY += blockSize {
		centerIdxX++
		for centerX := S / 2; centerX < width; centerX += blockSize {
			if centerX+S/2 > width || centerY+S/2 > height {
				continue
			}
			centerIdxY++
			point := Point{X: centerX, Y: centerY}

			centerPoint := CenterPoint{
				indexX:  centerIdxX,
				indexY:  centerIdxY,
				weights: make([][]float64, numberYOfBlock),
			}
			for i := 0; i < DescriptorParam_M; i++ {
				for j := 0; j < DescriptorParam_M; j++ {

				}
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
		_ = calculateCenters(w, h, float64(S_0))
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
