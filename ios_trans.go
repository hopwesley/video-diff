package main

import (
	"encoding/json"
	"fmt"
	"gocv.io/x/gocv"
	"io"
	"os"
)

func SimpleSpatial() {
	video, err := gocv.VideoCaptureFile(param.rawAFile)
	if err != nil {
		panic(err)
	}

	var frame = gocv.NewMat()
	if ok := video.Read(&frame); !ok || frame.Empty() {
		fmt.Println("Error reading video")
		frame.Close()
		return
	}
	var grayFrame = gocv.NewMat()
	gocv.CvtColor(frame, &grayFrame, gocv.ColorRGBToGray)
	frame.Close()
	grayFloat, err := matToFloatArray(grayFrame)
	if err != nil {
		panic(err)
	}
	saveMatAsImage(grayFrame, "simple_gray")
	saveJson("tmp/simple_gray.json", grayFloat)

	gradX := gocv.NewMat()
	gradY := gocv.NewMat()

	gocv.Sobel(grayFrame, &gradX, gocv.MatTypeCV16S, 1, 0, 3, 1, 0, gocv.BorderDefault)
	gocv.Sobel(grayFrame, &gradY, gocv.MatTypeCV16S, 0, 1, 3, 1, 0, gocv.BorderDefault)
	saveMatAsImage(gradX, "simple_grad_x")
	saveMatAsImage(gradY, "simple_grad_y")

	gradXFloat, err := matToFloatArray(gradX)
	if err != nil {
		panic(err)
	}
	saveJson("tmp/simple_grad_x.json", gradXFloat)
	//fmt.Println("gradXFloat:", gradXFloat)
	gradYFloat, err := matToFloatArray(gradY)
	if err != nil {
		panic(err)
	}
	//fmt.Println("gradYFloat:", gradYFloat)
	saveJson("tmp/simple_grad_y.json", gradYFloat)
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

func ReadIosFile() {
	file, err := os.Open("tmp/grayBuffer.json")
	if err != nil {
		fmt.Println("Error opening file:", err)
		return
	}
	defer file.Close()

	byteValue, err := io.ReadAll(file)
	if err != nil {
		fmt.Println("Error reading file:", err)
		return
	}

	var grayValues [][]uint8
	err = json.Unmarshal(byteValue, &grayValues)
	if err != nil {
		fmt.Println("Error unmarshalling JSON:", err)
		return
	}
	saveGrayDataData(grayValues, "tmp/grayBuffer.png")
}
