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
	Version         = "0.1.1"
	BaseSizeOfPixel = 32
	MaxSizeLevel    = 2
	Cell_M          = 2
	Cell_m          = 4
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

type Point struct {
	X float64
	Y float64
}

func init() {
	flags := rootCmd.Flags()
	flags.BoolVarP(&param.version, "version",
		"v", false, "golf -v")

	flags.StringVarP(&param.alignedAFile, "source",
		"a", "A.mp4", "golf -s A.mp4")

	flags.StringVarP(&param.alignedBFile, "dest",
		"b", "B.mp4", "golf -d B.mp4")

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

	procOneFrameForHistogram(gray, center, BaseSizeOfPixel)
	gray.Close()
}
func procOneFrameForHistogram(gray gocv.Mat, center Point, size int) [][]int {

	// 获取感兴趣的区域
	roi := getRegionOfInterest(gray, center, size)

	// 划分网格
	cells := divideIntoCells(roi, Cell_M)
	roi.Close()
	// 遍历每个小网格并计算直方图
	var hists [][]int
	for i, row := range cells {
		for j, cell := range row {
			hist := calculateHistogramForCell(cell)
			fmt.Printf("Cell [%d,%d] histogram: %v\n", i, j, hist)
			cell.Close() // 释放资源
			hists = append(hists, hist)
		}
	}
	return hists
}

func getRegionOfInterest(frame gocv.Mat, center Point, s int) gocv.Mat {
	x := int(center.X) - s/2
	y := int(center.Y) - s/2
	roi := frame.Region(image.Rect(x, y, x+s, y+s))
	return roi
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

func calculateHistogramForCell(cell gocv.Mat) []int {
	// 初始化梯度直方图数组

	// 对cell计算x和y方向的Sobel梯度
	gradX := gocv.NewMat()
	gradY := gocv.NewMat()

	defer gradX.Close()
	defer gradY.Close()
	gocv.Sobel(cell, &gradX, gocv.MatTypeCV16S, 1, 0, 3, 1, 0, gocv.BorderDefault)
	gocv.Sobel(cell, &gradY, gocv.MatTypeCV16S, 0, 1, 3, 1, 0, gocv.BorderDefault)

	return quantizeGradients(&gradX, &gradY, nil)
}
