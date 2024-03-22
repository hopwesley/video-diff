package main

import (
	"fmt"
	"github.com/spf13/cobra"
	"gocv.io/x/gocv"
	"image"
	"math"
)

type startParam struct {
	version      bool
	rawAFile     string
	rawBFile     string
	alignedAFile string
	alignedBFile string
	centerX      int
	centerY      int
	alignGap     int
}

const (
	Version          = "0.1.1"
	BaseSizeOfPixel  = 32
	Cell_M           = 2
	Cell_m           = 4
	SigmaForBaseSize = 6
	LevelOfDes       = 1
)

var rootCmd = &cobra.Command{
	Use: "golf",

	Short: "golf",

	Long: `usage description::TODO::`,

	Run: mainRun,
}
var (
	DebugFile          = true
	param              = &startParam{}
	icosahedronCenterP = generateIcosahedronFaces()
)

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
		faces[i], _ = normalize(face)
	}

	return faces
}
func init() {
	flags := rootCmd.Flags()
	flags.BoolVarP(&param.version, "version",
		"v", false, "golf -v")

	flags.StringVarP(&param.alignedAFile, "source",
		"a", "align_A.mp4", "golf -a align_A.mp4")

	flags.StringVarP(&param.alignedBFile, "dest",
		"b", "align_B.mp4", "golf -b align_B.mp4")

	flags.IntVarP(&param.centerX, "center-x", "x", -1, "")
	flags.IntVarP(&param.centerY, "center-y", "y", -1, "")

	rootCmd.AddCommand(alignCmd)
	rootCmd.AddCommand(testCmd)
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		panic(err)
	}
}

func mainRun(_ *cobra.Command, _ []string) {
	if param.version {
		fmt.Println(Version)
		return
	}
	fmt.Println("file name:", param.alignedAFile, param.alignedBFile)
	videoA, videoB, err := readFile(param.alignedAFile, param.alignedBFile)
	if err != nil {
		panic(err)
	}
	defer videoA.Close()
	defer videoB.Close()
	var idx = 0
	//for {
	idx++
	desOfA := procHistogram(videoA)
	if desOfA == nil {
		fmt.Println("video a finished")
		//break
	}
	desOfB := procHistogram(videoB)
	if desOfB == nil {
		fmt.Println("video b finished")
		//break
	}

	targetSize := (1 << LevelOfDes) * BaseSizeOfPixel
	finalMap := make([][]float64, targetSize)
	for i := range finalMap {
		finalMap[i] = make([]float64, targetSize)
	}

	for l := 0; l < LevelOfDes; l++ {
		w := wValueForOneLevel(desOfA[l], desOfB[l])
		weight := 1 << l
		wbi := bilinearInterpolate(w, weight*BaseSizeOfPixel)
		fmt.Println("w value with weight:", normalizeAndConvertToImage(wbi, fmt.Sprintf("tmp/wbi_layer_%d.png", l)))
		//for y := 0; y < targetSize; y++ {
		//	for x := 0; x < targetSize; x++ {
		//		finalMap[y][x] += weight * wbi[y][x]
		//	}
		//}
	}

	//}
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

func bilinearInterpolate(input [][]float64, outputSize int) [][]float64 {
	output := make([][]float64, outputSize)
	for i := range output {
		output[i] = make([]float64, outputSize)
	}

	scaleX := float64(len(input[0])) / float64(outputSize)
	scaleY := float64(len(input)) / float64(outputSize)

	for y := 0; y < outputSize; y++ {
		for x := 0; x < outputSize; x++ {
			srcX := float64(x) * scaleX
			srcY := float64(y) * scaleY

			x0 := int(math.Floor(srcX))
			x1 := x0 + 1
			if x1 >= len(input[0]) {
				x1 = len(input[0]) - 1
			}

			y0 := int(math.Floor(srcY))
			y1 := y0 + 1
			if y1 >= len(input) {
				y1 = len(input) - 1
			}

			fracX := srcX - float64(x0)
			fracY := srcY - float64(y0)

			p0 := input[y0][x0]*(1-fracX) + input[y0][x1]*fracX
			p1 := input[y1][x0]*(1-fracX) + input[y1][x1]*fracX

			output[y][x] = p0*(1-fracY) + p1*fracY
		}
	}

	return output
}

func wValueForOneLevel(desAOneLevel, desBOneLevel [][]float64) [][]float64 {
	// Assuming that desA and desB are [M][m*10] arrays where each cell contains m*10 blocks.
	// Initialize wForLevel with the same structure as desA and desB.
	wForLevel := make([][]float64, len(desAOneLevel))
	for i := range wForLevel {
		wForLevel[i] = make([]float64, len(desAOneLevel[i])/10)
	}

	// Iterate through each cell and block, computing the dissimilarity for each.
	for i, cellA := range desAOneLevel {
		cellB := desBOneLevel[i]
		for j := 0; j < len(cellA); j += 10 {
			// Here we assume that each block is represented by 10 histogram bins.
			histA := cellA[j : j+10]
			histB := cellB[j : j+10]
			ed := euclideanDistance(histA, histB)
			wForLevel[i][j/10] = ed // Store dissimilarity for each block
		}
	}

	// Return the dissimilarity matrix for the level
	return wForLevel
}

// euclideanDistance computes the Euclidean distance between two vectors.
func euclideanDistance(vec1, vec2 []float64) float64 {
	sum := 0.0
	for i := range vec1 {
		diff := vec1[i] - vec2[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

func procHistogram(video *gocv.VideoCapture) [][][]float64 {
	width := video.Get(gocv.VideoCaptureFrameWidth)
	height := video.Get(gocv.VideoCaptureFrameHeight)
	var center = Point{float64(param.centerX), float64(param.centerY)}
	if param.centerX < 0 {
		center.X = width / 2
	}
	if param.centerY < 0 {
		center.Y = height / 2
	}
	var frame = gocv.NewMat()
	if ok := video.Read(&frame); !ok || frame.Empty() {
		frame.Close()
		return nil
	}

	// 转换为灰度图
	gray := gocv.NewMat()
	gocv.CvtColor(frame, &gray, gocv.ColorBGRToGray)
	frame.Close()
	Descriptor := make([][][]float64, LevelOfDes)
	for l := 0; l < LevelOfDes; l++ {
		timer := 1 << l
		var sideOfRoi = timer * BaseSizeOfPixel
		histogramForFrame := procOneFrameForHistogram(gray, center, sideOfRoi, float64(sideOfRoi/SigmaForBaseSize))
		Descriptor[l] = histogramForFrame
	}

	gray.Close()
	return Descriptor
}

func procOneFrameForHistogram(gray gocv.Mat, center Point, size int, sigma float64) [][]float64 {
	fmt.Println("center of interest:", center)
	// 获取感兴趣的区域
	roiCenter, roi := getRegionOfInterest(gray, center, size)
	saveMatAsImage(roi, "roi/roi")
	fmt.Println("type of roi :=>", roi.Type())
	// 划分网格
	cells := divideIntoCells(roi, Cell_M)
	roi.Close()
	// 遍历每个小网格并计算直方图
	var hists [][]float64
	cellSize := size / Cell_M
	for i, row := range cells {
		for j, cell := range row {

			topLeftX := float64(j * cellSize)
			topLeftY := float64(i * cellSize)

			topLeftOfCell := Point{X: topLeftX, Y: topLeftY}
			saveMatAsImage(cell, "cell/cell_"+topLeftOfCell.String())

			hist := calculateHistogramForCell(cell, Cell_m, topLeftOfCell, roiCenter, sigma)
			cell.Close() // 释放资源
			hists = append(hists, hist)
		}
	}

	return normalizeHists(hists)
}

func normalizeHists(hists [][]float64) [][]float64 {
	normalizedHists := make([][]float64, len(hists))
	for i, hist := range hists {
		var norm float64
		for _, val := range hist {
			norm += val * val
		}
		norm = math.Sqrt(norm) + 1 // 计算L2范数并加1

		normalizedHists[i] = make([]float64, len(hist))
		for j, val := range hist {
			normalizedHists[i][j] = val / norm // 归一化处理
		}
	}

	return normalizedHists
}
func getRegionOfInterest(frame gocv.Mat, center Point, s int) (Point, gocv.Mat) {
	x := int(center.X) - s/2
	y := int(center.Y) - s/2
	roi := frame.Region(image.Rect(x, y, x+s, y+s))

	// 返回ROI的新中心坐标（相对于ROI的左上角），这里是ROI尺寸的一半，因为ROI是以中心为原点
	return Point{X: float64(s / 2), Y: float64(s / 2)}, roi
}

func divideIntoCells(roi gocv.Mat, M int) [][]gocv.Mat {
	cellSize := roi.Rows() / M // 或 roi.Cols() / M，因为是正方形区域
	cells := make([][]gocv.Mat, M)
	for i := range cells {
		cells[i] = make([]gocv.Mat, M)
		for j := range cells[i] {
			x := j * cellSize
			y := i * cellSize
			cells[i][j] = roi.Region(image.Rect(x, y, x+cellSize, y+cellSize))
		}
	}
	return cells
}

// 这个函数的目标是量化cell中的梯度，并计算加权直方图。
func calculateHistogramForCell(cell gocv.Mat, m int, topLeftOfCell, centerOfRoi Point, sigma float64) []float64 {
	cellSize := cell.Rows()
	blockSize := cellSize / m               // 获取小块的大小
	weightedHist := make([]float64, 10*m*m) // 初始化加权直方图数组

	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			// 计算小块的中心坐标
			blockX := topLeftOfCell.X + float64(j*blockSize) // 这里cellSize已经是小block的尺寸
			blockY := topLeftOfCell.Y + float64(i*blockSize) // 同上
			// 计算block的中心坐标
			centerOfBlock := Point{
				X: blockX + float64(blockSize/2),
				Y: blockY + float64(blockSize/2),
			}
			//fmt.Printf("\n center of block[row:%d, block:%d]: center:%s\n", i, j, centerOfBlock)

			// 提取小块
			block := cell.Region(image.Rect(j*blockSize, i*blockSize, (j+1)*blockSize, (i+1)*blockSize))
			saveMatAsImage(block, "block/block_"+centerOfBlock.String())
			// 计算小块的直方图
			blockHist := quantizeBlockGradients(block)
			// 获取高斯权重
			weight := centerOfBlock.GaussianKernel(centerOfRoi, sigma)
			// 加权直方图
			fmt.Println()
			for k, val := range blockHist {
				idx := (i*m+j)*10 + k
				weightedHist[idx] += float64(val) * weight
				fmt.Printf("\nblock histrogram i:%d j:%d idx:%d val:%d, weight:%f result:%f", i, j, idx, val, weight, weightedHist[idx])

			}
			fmt.Println()
			block.Close()
		}
	}
	// 正则化直方图
	return weightedHist
}

// 量化块内的梯度
func quantizeBlockGradients(block gocv.Mat) []int {
	gradX := gocv.NewMat()
	defer gradX.Close()
	gradY := gocv.NewMat()
	defer gradY.Close()
	gocv.Sobel(block, &gradX, gocv.MatTypeCV16S, 1, 0, 3, 1, 0, gocv.BorderDefault)
	gocv.Sobel(block, &gradY, gocv.MatTypeCV16S, 0, 1, 3, 1, 0, gocv.BorderDefault)
	return quantizeGradients(&gradX, &gradY, nil)
}
