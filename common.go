package main

import (
	"encoding/json"
	"fmt"
	"gocv.io/x/gocv"
	"io"
	"math"
	"os"
)

type Point struct {
	X float64
	Y float64
}

func (p Point) Distance(center Point) Point {
	return Point{
		X: p.X - center.X,
		Y: p.Y - center.Y,
	}
}
func (p Point) String() string {
	return fmt.Sprintf("{X:%.2f,Y:%.2f}", p.X, p.Y)
}
func (p Point) GaussianKernel(center Point, sigma float64) float64 {
	dx := p.X - center.X
	dy := p.Y - center.Y
	d := dx*dx + dy*dy
	return math.Exp(-d / (2 * sigma * sigma))
}

func normalize(vertex [3]float64) ([3]float64, bool) {
	length := math.Sqrt(vertex[0]*vertex[0] + vertex[1]*vertex[1] + vertex[2]*vertex[2])
	if length == 0 {
		return [3]float64{0.0, 0.0, 0.0}, true
	}
	return [3]float64{vertex[0] / length, vertex[1] / length, vertex[2] / length}, false
}

func saveVideoFromFrame(videoCapture *gocv.VideoCapture, startFrameIndex int, outputFile string) {
	// 设置视频捕获的位置
	videoCapture.Set(gocv.VideoCapturePosFrames, float64(startFrameIndex))

	// 获取视频的FPS和分辨率，以便于VideoWriter使用
	fps := videoCapture.Get(gocv.VideoCaptureFPS)
	width := int(videoCapture.Get(gocv.VideoCaptureFrameWidth))
	height := int(videoCapture.Get(gocv.VideoCaptureFrameHeight))

	// 初始化VideoWriter
	writer, err := gocv.VideoWriterFile(outputFile, "mp4v", fps, width, height, true)
	if err != nil {
		fmt.Println("Error initializing video writer:", err)
		return
	}
	defer writer.Close()

	// 读取并写入帧
	mat := gocv.NewMat()
	defer mat.Close()
	for {
		if ok := videoCapture.Read(&mat); !ok || mat.Empty() {
			break
		}
		writer.Write(mat)
	}
	fmt.Println("Video saved to:", outputFile)
}

func readFile(aFile, bFile string) (*gocv.VideoCapture, *gocv.VideoCapture, error) {
	av, err := gocv.VideoCaptureFile(aFile)
	if err != nil {
		return nil, nil, err
	}
	logVideoInfo(av)
	bv, err := gocv.VideoCaptureFile(bFile)
	if err != nil {
		return nil, nil, err
	}
	logVideoInfo(bv)
	return av, bv, nil
}

func logVideoInfo(video *gocv.VideoCapture) {

	width := video.Get(gocv.VideoCaptureFrameWidth)
	height := video.Get(gocv.VideoCaptureFrameHeight)
	fps := video.Get(gocv.VideoCaptureFPS)
	frameCount := video.Get(gocv.VideoCaptureFrameCount)

	fmt.Printf("Video Properties:\n")
	fmt.Printf("Width: %v\n", width)
	fmt.Printf("Height: %v\n", height)
	fmt.Printf("FPS: %v\n", fps)
	fmt.Printf("Total Frames: %v\n", frameCount)
}

// 计算两个向量的点积
func dotProduct(a, b [3]float64) float64 {
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

func projectGradient(gradient, faceCenter [3]float64) float64 {
	// 计算两个向量的点积
	return dotProduct(gradient, faceCenter)
}

// 该函数将被用于量化梯度函数中
func quantizeGradients(gradX, gradY, gradT *gocv.Mat) []int {
	histogram := make([]int, 20)
	fmt.Println(gradX.Type(), gradY.Type(), gradT.Type())
	for row := 0; row < gradX.Rows(); row++ {
		for col := 0; col < gradX.Cols(); col++ {
			// 获取梯度向量
			gx, gy := gradX.GetShortAt(row, col), gradY.GetShortAt(row, col)
			var gt = uint8(0)
			if gradT != nil {
				gt = gradT.GetUCharAt(row, col)
			}
			//fmt.Println("gt=>", gx, gy, gt)
			gradient, isZero := normalize([3]float64{float64(gx), float64(gy), float64(gt)})
			if isZero {
				//fmt.Println("this is zero val=>", row, col)
				continue
			}

			// 初始化变量以找到最大的点积值
			maxProjection := math.Inf(-1)
			maxIndex := -1
			projection := 0.0
			// 计算每个面中心的投影
			for i, faceCenter := range icosahedronCenterP {
				projection = projectGradient(gradient, faceCenter)
				// 更新最大点积值和索引
				fmt.Println("projection=>", projection)
				if projection > maxProjection {
					maxProjection = projection
					maxIndex = i
				}
			}
			//if maxProjection < threshold {
			//	fmt.Println("little maxProjection=>", maxProjection)
			//}
			// 在直方图中增加最接近的面中心位置的bin的计数
			if maxIndex >= 0 {
				fmt.Println(" maxProjection=>", maxProjection)
				histogram[maxIndex]++
			}
		}
	}

	// 现在需要将有方向的20-bin直方图合并为无方向的10-bin直方图
	return convertToUndirectedHistogram(histogram)
}

func quantizeGradients2(gradX, gradY, gradT *gocv.Mat) [][][]float64 {
	histogram := make([][][]float64, gradX.Rows())
	//fmt.Println(gradX.Type(), gradY.Type(), gradT.Type(), gradX.Rows(), gradX.Cols())
	for row := 0; row < gradX.Rows(); row++ {
		histogram[row] = make([][]float64, gradX.Cols())
		for col := 0; col < gradX.Cols(); col++ {
			histogram[row][col] = make([]float64, 10)
			// 获取梯度向量
			gx, gy := gradX.GetShortAt(row, col), gradY.GetShortAt(row, col)
			var gt = uint8(0)
			if gradT != nil {
				gt = gradT.GetUCharAt(row, col)
			}

			//fmt.Println("gt=>", gx, gy, gt)
			gradient := [3]float64{float64(gx), float64(gy), float64(gt)}
			gradientL2 := norm2Float(gradient[:])
			if gradientL2 == 0.0 {
				continue
			}

			gradient[0] = gradient[0] / gradientL2
			gradient[1] = gradient[1] / gradientL2
			gradient[2] = gradient[2] / gradientL2

			// 合并对立方向的梯度值
			for i := 0; i < 10; i++ {
				pi, pi10 := projectGradient(gradient, icosahedronCenterP[i]), projectGradient(gradient, icosahedronCenterP[i+10])
				onePos := math.Abs(pi)
				twoPos := math.Abs(pi10)
				histogram[row][col][i] = onePos + twoPos - threshold
				if histogram[row][col][i] < 0 {
					histogram[row][col][i] = 0.0
				}
				//fmt.Println("(row,col)=>[i:(i,i+10,sum(i)]=>", row, col, i, pi, pi10, onePos, twoPos, project[i])
			}
			pL2 := norm2Float(histogram[row][col])
			if pL2 == 0.0 {
				continue
			}
			for i := 0; i < 10; i++ {
				histogram[row][col][i] = histogram[row][col][i] / pL2 * gradientL2
				//fmt.Println(project[i])
			}
		}
	}
	return histogram
}

func convertToUndirectedHistogram(directedHistogram []int) []int {
	undirectedHistogram := make([]int, 10)
	for i := 0; i < 10; i++ {
		undirectedHistogram[i] = directedHistogram[i] + directedHistogram[i+10]
	}
	return undirectedHistogram
}

func calculateNCC(histogramA, histogramB []float64) float64 {
	meanA := calculateMean(histogramA)
	meanB := calculateMean(histogramB)

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

func calculateMean(histogram []float64) float64 {
	sum := 0.0
	for _, value := range histogram {
		sum += value
	}
	return sum / float64(len(histogram))
}

// 计算2范数
func norm2Float(hist []float64) float64 {
	sum := 0.0
	for _, value := range hist {
		sum += value * value
	}
	return math.Sqrt(sum)
}

// 计算2范数
func norm2(hist []int) float64 {
	sum := 0.0
	for _, value := range hist {
		sum += float64(value * value)
	}
	return math.Sqrt(sum)
}

var tmpIdx = 0

func saveMatAsImage(mat gocv.Mat, filename string) bool {
	if !DebugFile {
		return true
	}
	filename = fmt.Sprintf("tmp/%s_%d.png", filename, tmpIdx)
	tmpIdx++

	// 将16位的图像转换为8位
	return __saveImg(mat, filename)
}

func __saveImg(mat gocv.Mat, filename string) bool {
	converted := gocv.NewMat()
	defer converted.Close()

	minVal, maxVal, _, _ := gocv.MinMaxIdx(mat)

	// 中间值为最小值和最大值的平均
	midVal := (minVal + maxVal) / 2.0

	// 计算 alpha 和 beta
	// 使得 midVal 映射到 128, maxVal 映射到 255, minVal 映射到 0
	alpha := 255.0 / (maxVal - minVal)
	beta := 128 - midVal*alpha

	mat.ConvertToWithParams(&converted, gocv.MatTypeCV8U, alpha, beta)
	//mat.ConvertTo(&converted, gocv.MatTypeCV8U)
	// 写入文件
	return gocv.IMWrite(filename, converted)
}

func __saveNormalizedData(data [][]float64, fileName string) bool {
	height := len(data)
	width := len(data[0])
	img := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8U)
	for y, row := range data {
		for x, val := range row {
			grayVal := uint8(val * 255) // 将值映射到0到255
			img.SetUCharAt(y, x, grayVal)
		}
	}
	return gocv.IMWrite(fileName, img)
}

func saveGrayDataData(data [][]uint8, fileName string) bool {
	height := len(data)
	width := len(data[0])
	img := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8U)
	for y, row := range data {
		for x, val := range row {
			img.SetUCharAt(y, x, val)
		}
	}
	return gocv.IMWrite(fileName, img)
}

func saveJson(fileName string, data any) {
	file, _ := os.Create(fileName)
	dataBytes, _ := json.Marshal(data)
	file.Write(dataBytes)
	file.Close()
}

func readJson(fileName string, data any) error {
	// 打开文件
	file, err := os.Open(fileName)
	if err != nil {
		return err
	}
	defer file.Close()

	// 读取文件内容到字节切片
	dataBytes, err := io.ReadAll(file)
	if err != nil {
		return err
	}

	// 定义一个接收数据的变量

	// 解码JSON数据到预定义的结构
	err = json.Unmarshal(dataBytes, data)
	if err != nil {
		return err
	}

	return nil
}

func normalizeAndConvertToImage(wbi [][]float64, filename string) bool {
	height := len(wbi)
	width := len(wbi[0])
	img := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8U)

	// 找到wbi中的最大值和最小值
	maxVal := wbi[0][0]
	minVal := wbi[0][0]
	for _, row := range wbi {
		for _, val := range row {
			if val > maxVal {
				maxVal = val
			}
			if val < minVal {
				minVal = val
			}
		}
	}

	// 确保最大值不是0，避免除以0的情况
	if maxVal == 0 {
		maxVal = 1
	}

	// 归一化并将浮点数转换为uint8类型的灰度值
	for y, row := range wbi {
		for x, val := range row {
			normalizedVal := (val - minVal) / (maxVal - minVal) // 将值归一化到0到1
			grayVal := uint8(normalizedVal * 255)               // 将值映射到0到255
			img.SetUCharAt(y, x, grayVal)
		}
	}

	return gocv.IMWrite(filename, img)
}

func normalizeImage(wbi [][]float64) [][]float64 {

	// 找到wbi中的最大值和最小值
	maxVal := wbi[0][0]
	minVal := wbi[0][0]
	for _, row := range wbi {
		for _, val := range row {
			if val > maxVal {
				maxVal = val
			}
			if val < minVal {
				minVal = val
			}
		}
	}
	//fmt.Println("maxVal------>>>:", maxVal)
	if maxVal == 0 {
		maxVal = 1
	}
	normalizedWbi := make([][]float64, len(wbi))
	for y, row := range wbi {
		normalizedWbi[y] = make([]float64, len(wbi[0]))
		for x, val := range row {

			normalizedVal := (val - minVal) / (maxVal - minVal) // 将值归一化到0到1
			normalizedWbi[y][x] = normalizedVal
		}
	}

	return normalizedWbi
}

func DataToImage(fileName string, grayValues [][]uint8) {
	file, err := os.Open(fileName)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	byteValue, err := io.ReadAll(file)
	if err != nil {
		panic(err)

	}
	err = json.Unmarshal(byteValue, &grayValues)
	if err != nil {
		panic(err)
	}
	saveGrayDataData(grayValues, fileName+".png")
}
