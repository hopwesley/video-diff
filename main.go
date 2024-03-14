package main

import (
	"fmt"
	"github.com/spf13/cobra"
	"gocv.io/x/gocv"
	"math"
)

type startParam struct {
	version bool
	aFile   string
	bFile   string
}

var rootCmd = &cobra.Command{
	Use: "golf",

	Short: "golf",

	Long: `usage description::TODO::`,

	Run: mainRun,
}
var (
	param       = &startParam{}
	faceCenters = generateIcosahedronFaces()
)

func normalize(vertex [3]float64) [3]float64 {
	length := math.Sqrt(vertex[0]*vertex[0] + vertex[1]*vertex[1] + vertex[2]*vertex[2])
	return [3]float64{vertex[0] / length, vertex[1] / length, vertex[2] / length}
}

// 定义黄金分割比
var phi = (1.0 + math.Sqrt(5.0)) / 2.0

func generateIcosahedronFaces() [][3]float64 {

	// 这些是根据论文中给出的坐标定义的正二十面体的面的中心位置
	faces := [][3]float64{
		{0, 1 / phi, phi}, {0, -1 / phi, phi},
		{0, 1 / phi, -phi}, {0, -1 / phi, -phi},
		{1 / phi, phi, 0}, {-1 / phi, phi, 0},
		{1 / phi, -phi, 0}, {-1 / phi, -phi, 0},
		{phi, 0, 1 / phi}, {-phi, 0, 1 / phi},
		{phi, 0, -1 / phi}, {-phi, 0, -1 / phi},
		{1, 1, 1}, {-1, 1, 1},
		{1, -1, 1}, {-1, -1, 1},
		{1, 1, -1}, {-1, 1, -1},
		{1, -1, -1}, {-1, -1, -1},
	}

	// 标准化每个面的中心位置
	for i, face := range faces {
		faces[i] = normalize(face)
	}

	return faces
}

// 计算两个向量的点积
func dotProduct(a, b [3]float64) float64 {
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

func init() {
	flags := rootCmd.Flags()
	flags.BoolVarP(&param.version, "version",
		"v", false, "golf -v")

	flags.StringVarP(&param.aFile, "source",
		"s", "A.mp4", "golf -s A.mp4")

	flags.StringVarP(&param.bFile, "dest",
		"d", "B.mp4", "golf -d B.mp4")
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		panic(err)
	}
}

const (
	PercentForMaxDepthToTimeAlign = 0.2 //20%of all frames to find the time start
)

func mainRun(_ *cobra.Command, _ []string) {

	videoA, videoB, err := readFile(param.aFile, param.bFile)
	if err != nil {
		panic(fmt.Errorf("failed tor read video file %s", param.aFile))
	}
	defer videoA.Close()
	defer videoB.Close()

	aHisGram, _ := parseHistogram(videoA)
	bHisGram, _ := parseHistogram(videoB)

	// 应用阈值处理
	threshold := 1.29107 // 根据论文描述的阈值

	aHisGramFloat := distributeGradientMagnitude(aHisGram, threshold)
	bHisGramFloat := distributeGradientMagnitude(bHisGram, threshold)

	idxA, idxB := findTimeStartOfFrame(aHisGramFloat, bHisGramFloat)

	if idxB < 0 || idxA < 0 {
		panic("find time start frame failed")
	}
	fmt.Println("time align =>", idxA, idxB)
	var startA, startB = 0, 0
	if idxA > idxB {
		startA = idxA - idxB
		startB = 0
	} else {
		startB = idxB - idxA
		startA = 0
	}
	fmt.Println("time align =>", startA, startB)

	saveVideoFromFrame(videoA, startA, "align_"+param.aFile)
	saveVideoFromFrame(videoB, startB, "align_"+param.bFile)
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

func findTimeStartOfFrame(aHisGramFloat, bHisGramFloat [][]float64) (int, int) {

	var maxDepth = 0
	var longGram [][]float64
	var shortGram [][]float64
	if len(aHisGramFloat) < len(bHisGramFloat) {
		maxDepth = len(bHisGramFloat)
		longGram = bHisGramFloat
		shortGram = aHisGramFloat
	} else {
		maxDepth = len(aHisGramFloat)
		longGram = aHisGramFloat
		shortGram = bHisGramFloat
	}
	maxDepth = int(float32(maxDepth) * PercentForMaxDepthToTimeAlign)

	videoALength := len(longGram)  // Video A frame count
	videoBLength := len(shortGram) // Video B frame count

	// Initialize a 2D array to store the NCC values
	nccValues := make([][]float64, videoALength)
	for i := range nccValues {
		nccValues[i] = make([]float64, videoBLength)
	}

	// Iterate over all frame pairs of Video A and Video B, calculate their NCC values
	for i, histogramA := range longGram {
		if i > maxDepth {
			break
		}
		for j, histogramB := range shortGram {
			nccValues[i][j] = calculateNCC(histogramA, histogramB)
		}
	}

	maxNCC := -1.0       // Assuming NCC values range from -1 to 1, start with the minimum possible value
	maxI, maxJ := -1, -1 // To store the indices of the maximum NCC value

	// Find the maximum NCC value and its corresponding indices
	for i, row := range nccValues {
		for j, nccValue := range row {
			if nccValue > maxNCC {
				maxNCC = nccValue // Update the maximum NCC value
				maxI, maxJ = i, j // Update the indices of the maximum NCC value
			}
		}
	}

	return maxI, maxJ // These are the indices of the frames that best align in time
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

func parseHistogram(video *gocv.VideoCapture) ([][]int, error) {

	// 初始化前一帧变量
	var prevFrame gocv.Mat
	firstFrame := true

	var histograms [][]int // 用于存储每一帧的直方图

	for {
		var frame = gocv.NewMat()
		if ok := video.Read(&frame); !ok || frame.Empty() {
			frame.Close()
			break
		}
		// Convert to grayscale
		var grayFrame = gocv.NewMat()
		gocv.CvtColor(frame, &grayFrame, gocv.ColorBGRToGray)
		frame.Close()
		if firstFrame {
			firstFrame = false
			prevFrame = grayFrame.Clone()
			continue
		}

		// Calculate spatial gradients
		gradX := gocv.NewMat()
		gradY := gocv.NewMat()
		gradT := gocv.NewMat()

		gocv.Sobel(grayFrame, &gradX, gocv.MatTypeCV16S, 1, 0, 3, 1, 0, gocv.BorderDefault)
		gocv.Sobel(grayFrame, &gradY, gocv.MatTypeCV16S, 0, 1, 3, 1, 0, gocv.BorderDefault)
		gocv.AbsDiff(grayFrame, prevFrame, &gradT)

		prevFrame.Close()
		// Make the current frame the new previous frame for the next iteration
		prevFrame = grayFrame.Clone()

		// Quantize gradients into a histogram using an icosahedron
		histogram := quantizeGradients(gradX, gradY, gradT)
		histograms = append(histograms, histogram) // 将当前帧的直方图添加到数组中

		gradX.Close()
		gradY.Close()
		gradT.Close()
		grayFrame.Close()
	}

	// Release the last previous frame
	if !prevFrame.Empty() {
		prevFrame.Close()
	}
	return histograms, nil // 返回包含每一帧直方图的数组
}

func projectGradient(gradient, faceCenter [3]float64) float64 {
	// 计算两个向量的点积
	return dotProduct(gradient, faceCenter)
}

// 该函数将被用于量化梯度函数中
func quantizeGradients(gradX, gradY, gradT gocv.Mat) []int {
	histogram := make([]int, 20)

	gradTFloat := gocv.NewMat()
	gradT.ConvertTo(&gradTFloat, gocv.MatTypeCV32F) // 转换为32位浮点类型
	defer gradTFloat.Close()

	for row := 0; row < gradX.Rows(); row++ {
		for col := 0; col < gradX.Cols(); col++ {
			// 获取梯度向量
			gx, gy, gt := gradX.GetFloatAt(row, col), gradY.GetFloatAt(row, col), gradTFloat.GetFloatAt(row, col)
			gradient := normalize([3]float64{float64(gx), float64(gy), float64(gt)})

			// 初始化变量以找到最大的点积值
			maxProjection := math.Inf(-1)
			maxIndex := -1

			// 计算每个面中心的投影
			for i, faceCenter := range faceCenters {
				projection := projectGradient(gradient, faceCenter)
				// 更新最大点积值和索引
				if projection > maxProjection {
					maxProjection = projection
					maxIndex = i
				}
			}

			// 在直方图中增加最接近的面中心位置的bin的计数
			if maxIndex >= 0 {
				histogram[maxIndex]++
			}
		}
	}

	// 现在需要将有方向的20-bin直方图合并为无方向的10-bin直方图
	return convertToUndirectedHistogram(histogram)
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
		numerator += (float64(histogramA[i]) - meanA) * (float64(histogramB[i]) - meanB)
		denominatorA += (float64(histogramA[i]) - meanA) * (float64(histogramA[i]) - meanA)
		denominatorB += (float64(histogramB[i]) - meanB) * (float64(histogramB[i]) - meanB)
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
func norm2(hist []int) float64 {
	sum := 0.0
	for _, value := range hist {
		sum += float64(value * value)
	}
	return math.Sqrt(sum)
}

// 分配梯度幅度和应用阈值
func distributeGradientMagnitude(hists [][]int, threshold float64) [][]float64 {
	processedHists := make([][]float64, len(hists))

	for i, hist := range hists {
		// 计算原始直方图的范数 (gNorm)。
		gNorm := norm2(hist)

		// 应用阈值处理。
		qPrime := make([]float64, len(hist))
		sumSq := 0.0 // 这将用于存储 qPrime 的平方和。
		for j, value := range hist {
			newValue := float64(value) - threshold
			if newValue < 0 {
				newValue = 0
			}
			qPrime[j] = newValue
			sumSq += newValue * newValue
		}

		// 计算 qPrime 的范数。
		qPrimeNorm := math.Sqrt(sumSq)

		// 计算 q，使用 gNorm 乘以 qPrime 的每个元素。
		processedHists[i] = make([]float64, len(hist))
		for j, qPrimeValue := range qPrime {
			if qPrimeNorm == 0 {
				processedHists[i][j] = 0
			} else {
				processedHists[i][j] = (gNorm * qPrimeValue) / qPrimeNorm
			}
		}
	}

	return processedHists
}
