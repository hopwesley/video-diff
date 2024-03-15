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
}

const (
	Version          = "0.1.1"
	BaseSizeOfPixel  = 32
	MaxSizeLevel     = 2
	Cell_M           = 2
	Cell_m           = 4
	SigmaForBaseSize = 6
)

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
	videoA, videoB, err := readFile(param.alignedAFile, param.alignedBFile)
	if err != nil {
		panic(err)
	}
	defer videoA.Close()
	defer videoB.Close()
	procHistogram(videoA)
	procHistogram(videoB)
}

func procHistogram(video *gocv.VideoCapture) {
	width := video.Get(gocv.VideoCaptureFrameWidth)
	height := video.Get(gocv.VideoCaptureFrameHeight)
	var center Point
	if param.centerX < 0 {
		center.X = width / 2
	}
	if param.centerY < 0 {
		center.Y = height / 2
	}
	var frame = gocv.NewMat()
	if ok := video.Read(&frame); !ok || frame.Empty() {
		frame.Close()
		return
	}

	// 转换为灰度图
	gray := gocv.NewMat()
	gocv.CvtColor(frame, &gray, gocv.ColorBGRToGray)
	frame.Close()

	histogramForFrame := procOneFrameForHistogram(gray, center, BaseSizeOfPixel, SigmaForBaseSize)
	fmt.Println("histogram for one frame:=>", histogramForFrame)
	gray.Close()
}

func procOneFrameForHistogram(gray gocv.Mat, center Point, size int, sigma float64) [][]float64 {

	// 获取感兴趣的区域
	roiCenter, roi := getRegionOfInterest(gray, center, size)
	// 划分网格
	cells := divideIntoCells(roi, Cell_M)
	roi.Close()
	// 遍历每个小网格并计算直方图
	var hists [][]float64
	for i, row := range cells {
		for j, cell := range row {
			cellCenterInRoI := Point{
				X: float64(j*cell.Cols() + cell.Cols()/2),
				Y: float64(i*cell.Rows() + cell.Rows()/2),
			}
			hist := calculateHistogramForCell(cell, Cell_m, cellCenterInRoI, roiCenter, sigma)
			fmt.Printf("Cell [%d,%d] histogram: %v\n", i, j, hist)
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
			x := i * cellSize
			y := j * cellSize
			cells[i][j] = roi.Region(image.Rect(x, y, x+cellSize, y+cellSize))
		}
	}
	return cells
}

// 这个函数的目标是量化cell中的梯度，并计算加权直方图。
func calculateHistogramForCell(cell gocv.Mat, m int, centerOfCell, centerOfRoi Point, sigma float64) []float64 {
	cellSize := cell.Rows() / m             // 获取小块的大小
	weightedHist := make([]float64, 10*m*m) // 初始化加权直方图数组

	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			// 计算小块的中心坐标
			centerOfBlock := Point{
				X: centerOfCell.X + float64(j*cellSize+cellSize/2),
				Y: centerOfCell.Y + float64(i*cellSize+cellSize/2),
			}
			// 提取小块
			block := cell.Region(image.Rect(j*cellSize, i*cellSize, (j+1)*cellSize, (i+1)*cellSize))
			// 计算小块的直方图
			blockHist := quantizeBlockGradients(block)
			// 获取高斯权重
			weight := centerOfBlock.GaussianKernel(centerOfRoi, sigma)
			// 加权直方图
			for k, val := range blockHist {
				weightedHist[i*m+j*10+k] += float64(val) * weight
			}
			block.Close()
		}
	}
	// 正则化直方图
	return weightedHist
}

// 量化块内的梯度
func quantizeBlockGradients(block gocv.Mat) []int {
	gradX := gocv.NewMat()
	gradY := gocv.NewMat()
	gocv.Sobel(block, &gradX, gocv.MatTypeCV16S, 1, 0, 3, 1, 0, gocv.BorderDefault)
	gocv.Sobel(block, &gradY, gocv.MatTypeCV16S, 0, 1, 3, 1, 0, gocv.BorderDefault)
	return quantizeGradients(&gradX, &gradY, nil)
}
