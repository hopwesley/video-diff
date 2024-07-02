package main

import (
	"fmt"
	"github.com/spf13/cobra"
	"gocv.io/x/gocv"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"sort"
	"time"
)

var testCmd = &cobra.Command{
	Use: "test",

	Short: "test",

	Long: `usage description::TODO::`,

	Run: testRun,
}

type testParam struct {
	op       int
	width    int
	height   int
	window   int
	cx       int
	cy       int
	cs       int
	alpha    float64
	betaLow  float64
	betaHigh float64
}

var testTool = &testParam{}

func init() {
	flags := testCmd.Flags()

	flags.IntVarP(&testTool.op, "operation",
		"o", 1, "golf test -o 1")

	flags.IntVarP(&testTool.width, "width",
		"x", 128, "golf test -x 720")
	flags.IntVarP(&testTool.height, "height",
		"y", 128, "golf test -y 1280")
	flags.IntVarP(&testTool.window, "window",
		"w", 30, "golf test -w 3")

	flags.StringVarP(&param.rawAFile, "source",
		"a", "A.mp4", "golf -a A.mp4")

	flags.StringVarP(&param.rawBFile, "dest",
		"b", "B.mp4", "golf -b B.mp4")

	flags.IntVarP(&testTool.cx, "centerX",
		"c", 32, "golf test -c 64")
	flags.IntVarP(&testTool.cy, "centerY",
		"d", 32, "golf test -d 64")
	flags.IntVarP(&testTool.cs, "size",
		"s", 32, "golf test -s 32")

	flags.StringVarP(&param.alignedAFile, "align-source",
		"k", "align_a.mp4", "golf -k align_a.mp4")

	flags.StringVarP(&param.alignedBFile, "align-dest",
		"l", "align_b.mp4", "golf -l align_b.mp4")

	flags.Float64VarP(&testTool.alpha, "alpha", "f", 0.75, "golf test -f 0.75")
	flags.Float64VarP(&testTool.betaLow, "betaL", "i", 0.2, "golf test -i 0.2")
	flags.Float64VarP(&testTool.betaHigh, "betaH", "j", 0.8, "golf test -j 0.8")
}

func testRun(_ *cobra.Command, _ []string) {
	switch testTool.op {
	case 1:
		createTestImg()
		return
	case 2:
		ComputeGradient()
		return
	case 3:
		ComputeNcc()
		return
	case 4:
		AlignVideo()
		return
	case 5:
		ComputeDesc()
		return
	case 6:
		ComputeWtl()
		return
	case 7:
		ComputeWt()
		return
	case 8:
		ComputeG()
		return
	case 9:
		AdjustContrast()
		return
	case 10:
		ComputeFC()
		return
	case 11:
		ComputeOverlay()
		return
	case 12:
		ComputeVideoDiff()
		return
	case 13:
		ComputeDiffImg()
		return

	case 14:
		ComputeG2()
		return

	case 15:
		AITest()
		return
	case 16:
		SimpleSpatial()
		return
	case 17:
		IosQuantizeGradient()
		return
	case 18:
		//testZeroFrameGradient()
		//testCpuOrGpu()
		//grayDataToImg("tmp/ios/gpu_grayBufferA.json")
		//grayDataToImg("tmp/ios/gpu_grayBufferB.json")
		//grayDataToImg("tmp/ios/gpu_gradientTBuffer.json")
		//histogramToImg("tmp/ios/gpu_frame_quantity_4.json")
		//gradientToImg("tmp/ios/gpu_gradientXBuffer.json")
		//gradientToImg("tmp/ios/gpu_gradientYBuffer.json")
		//histogramToImg("tmp/ios/cpu_block_gradient_one_frame_align_b.mp4_32.json")
		//histogramToImg("tmp/ios/cpu_block_gradient_one_frame_align_b.mp4_64.json")
		//histogramToImg("tmp/ios/cpu_block_gradient_one_frame_align_b.mp4_128.json")
		//
		//gradientToImg("tmp/ios/gpu_frame_histogram_A.json")
		//gradientToImg("tmp/ios/gpu_frame_histogram_B.json")
		//gradientToImg("tmp/ios/cpu_frame_histogram_A_1.mp4_32.json")
		//gradientToImg("tmp/ios/cpu_frame_histogram_B_1.mp4_32.json")

		return
	case 19:
		S_0 := 32
		AverageGradientOfBlock(S_0, param.alignedAFile)
		AverageGradientOfBlock(S_0, param.alignedAFile)
		return
	case 20:
		S_0 := 32
		now := time.Now().UnixMilli()
		//FrameQForTimeAlign(param.rawAFile, S_0)
		//FrameQForTimeAlign(param.rawBFile, S_0)

		FrameQForTimeAlign(param.alignedAFile, S_0)
		FrameQForTimeAlign(param.alignedBFile, S_0)

		fmt.Println("时长：", time.Now().UnixMilli()-now)
		return
	case 21:
		CommTest()
		return
	case 22:
		var S_0 = 32
		GradientOfBlockInOneFrame(S_0, param.alignedAFile)
		GradientOfBlockInOneFrame(S_0*2, param.alignedAFile)
		GradientOfBlockInOneFrame(S_0*4, param.alignedAFile)
		GradientOfBlockInOneFrame(S_0, param.alignedBFile)
		GradientOfBlockInOneFrame(S_0*2, param.alignedBFile)
		GradientOfBlockInOneFrame(S_0*4, param.alignedBFile)
		return
	case 23:
		var gradient [][]Histogram
		readJson("tmp/ios/cpu_block_gradient_one_frame_align_a.mp4_32.json", &gradient)
		DescriptorOfOneCenter(gradient, 32, weightsWithDistance[0])
		readJson("tmp/ios/cpu_block_gradient_one_frame_align_a.mp4_64.json", &gradient)
		DescriptorOfOneCenter(gradient, 64, weightsWithDistance[1])
		readJson("tmp/ios/cpu_block_gradient_one_frame_align_a.mp4_128.json", &gradient)
		DescriptorOfOneCenter(gradient, 128, weightsWithDistance[2])
		return
	case 24:
		var gradientA [][]Histogram
		var gradientB [][]Histogram
		readJson("tmp/ios/cpu_block_gradient_one_frame_align_a.mp4_32.json", &gradientA)
		readJson("tmp/ios/cpu_block_gradient_one_frame_align_b.mp4_32.json", &gradientB)
		DescOfOneFrame(gradientA, gradientB, 32, weightsWithDistance[0])
		readJson("tmp/ios/cpu_block_gradient_one_frame_align_a.mp4_64.json", &gradientA)
		readJson("tmp/ios/cpu_block_gradient_one_frame_align_b.mp4_64.json", &gradientB)
		DescOfOneFrame(gradientA, gradientB, 64, weightsWithDistance[1])
		readJson("tmp/ios/cpu_block_gradient_one_frame_align_a.mp4_128.json", &gradientA)
		readJson("tmp/ios/cpu_block_gradient_one_frame_align_b.mp4_128.json", &gradientB)
		DescOfOneFrame(gradientA, gradientB, 128, weightsWithDistance[2])
		return
	case 25:
		IosOldRoiHistogram()
		return

	case 26:
		var blockGradientA [][]Histogram
		var blockGradientB [][]Histogram
		readJson("tmp/ios/cpu_block_gradient_one_frame_align_a.mp4_32.json", &blockGradientA)
		readJson("tmp/ios/cpu_block_gradient_one_frame_align_b.mp4_32.json", &blockGradientB)
		WtlOfOneCenter(blockGradientA, blockGradientB, 10, 32, weightsWithDistance[0])
		WtlOfOneCenter(blockGradientA, blockGradientB, 11, 32, weightsWithDistance[0])
		readJson("tmp/ios/cpu_block_gradient_one_frame_align_a.mp4_64.json", &blockGradientA)
		readJson("tmp/ios/cpu_block_gradient_one_frame_align_b.mp4_64.json", &blockGradientB)
		WtlOfOneCenter(blockGradientA, blockGradientB, 10, 64, weightsWithDistance[1])
		WtlOfOneCenter(blockGradientA, blockGradientB, 11, 64, weightsWithDistance[1])
		readJson("tmp/ios/cpu_block_gradient_one_frame_align_a.mp4_128.json", &blockGradientA)
		readJson("tmp/ios/cpu_block_gradient_one_frame_align_b.mp4_128.json", &blockGradientB)
		WtlOfOneCenter(blockGradientA, blockGradientB, 10, 128, weightsWithDistance[2])
		WtlOfOneCenter(blockGradientA, blockGradientB, 11, 128, weightsWithDistance[2])
		return
	case 27:
		var blockGradientA [][]Histogram
		var blockGradientB [][]Histogram
		readJson("tmp/ios/cpu_block_gradient_one_frame_align_a.mp4_32.json", &blockGradientA)
		readJson("tmp/ios/cpu_block_gradient_one_frame_align_b.mp4_32.json", &blockGradientB)
		WtlOfOneFrame(blockGradientA, blockGradientB, 32, weightsWithDistance[0])
		readJson("tmp/ios/cpu_block_gradient_one_frame_align_a.mp4_64.json", &blockGradientA)
		readJson("tmp/ios/cpu_block_gradient_one_frame_align_b.mp4_64.json", &blockGradientB)
		WtlOfOneFrame(blockGradientA, blockGradientB, 64, weightsWithDistance[1])
		readJson("tmp/ios/cpu_block_gradient_one_frame_align_a.mp4_128.json", &blockGradientA)
		readJson("tmp/ios/cpu_block_gradient_one_frame_align_b.mp4_128.json", &blockGradientB)
		WtlOfOneFrame(blockGradientA, blockGradientB, 128, weightsWithDistance[2])
		return

	case 28:
		var S_0 = 32
		var sigma = float64(S_0 / Cell_M / Cell_m)
		calculateDistances(S_0, Cell_M, Cell_m, sigma)   //float64(S_0/4))
		calculateDistances(S_0*2, Cell_M, Cell_m, sigma) // float64(S_0/2))
		calculateDistances(S_0*4, Cell_M, Cell_m, sigma) //float64(S_0))
		return

	case 29:
		capA, err := gocv.VideoCaptureFile(param.alignedAFile)
		if err != nil {
			panic(err)
		}
		defer capA.Close()

		width := int(capA.Get(gocv.VideoCaptureFrameWidth))
		height := int(capA.Get(gocv.VideoCaptureFrameHeight))
		BiLinearInterpolate(width, height)
		return

	case 30:
		OverlayForOneFrame()
		return
	case 31:
		OverlayOneFrameFromStart()
		return
	case 32:
		FinalFantasy()
		return
	case 33:
		AlignVideoFromStart()
		return
	}
}

func gradient(grayFrame, prevFrame gocv.Mat) {
	gradX := gocv.NewMat()
	gradY := gocv.NewMat()
	gradT := gocv.NewMat()

	gocv.Sobel(grayFrame, &gradX, gocv.MatTypeCV16S, 1, 0, 3, 1, 0, gocv.BorderDefault)
	gocv.Sobel(grayFrame, &gradY, gocv.MatTypeCV16S, 0, 1, 3, 1, 0, gocv.BorderDefault)
	gocv.AbsDiff(grayFrame, prevFrame, &gradT)

	histogram := quantizeGradients2(&gradX, &gradY, &gradT)

	gradX.Close()
	gradY.Close()
	gradT.Close()

	fmt.Println(histogram)
}

func ComputeGradient() {
	frameA, frameA2, frameB, frameB2 := testData()

	defer frameA.Close()
	defer frameB.Close()
	defer frameA2.Close()
	defer frameB2.Close()

	gradient(frameA, frameA2)
	//gradient(frameB, frameB2)
}

func ComputeNcc() {
	videoA, videoB, err := readFile(param.rawAFile, param.rawBFile)
	if err != nil {
		panic(fmt.Errorf("failed tor read video file %s", param.rawAFile))
	}
	defer videoA.Close()
	defer videoB.Close()

	aHisGram, _ := parseHistogram2(videoA)
	bHisGram, _ := parseHistogram2(videoB)

	saveJson("a_histogram.txt", aHisGram)
	saveJson("b_histogram.txt", bHisGram)
}

func AlignVideo() {
	var aHisGram [][]float64
	var bHisGram [][]float64
	_ = readJson("a_histogram.txt", &aHisGram)
	_ = readJson("b_histogram.txt", &bHisGram)
	ncc := nccOfAllFrame(aHisGram, bHisGram)
	saveJson("ncc.txt", ncc)
	startA, startB, _ := findMaxNCCSequence(ncc, testTool.window)
	videoA, videoB, _ := readFile(param.rawAFile, param.rawBFile)
	saveVideoFromFrame(videoA, startA, "align_"+param.rawAFile)
	saveVideoFromFrame(videoB, startB, "align_"+param.rawBFile)

	fmt.Println("window:", testTool.window)
	fmt.Println("startA:", startA)
	fmt.Println("startB:", startB)
	fmt.Println(ncc[startA][startB])
}

func createTestImg() {
	frameA, frameA2, frameB, frameB2 := testData()
	defer frameA.Close()
	defer frameB.Close()
	defer frameA2.Close()
	defer frameB2.Close()

	__saveImg(frameA, "tmp/test/t_a.png")
	__saveImg(frameB, "tmp/test/t_b.png")
	__saveImg(frameA2, "tmp/test/t_a2.png")
	__saveImg(frameB2, "tmp/test/t_b2.png")
}

func testData() (frameA, frameA2, frameB, frameB2 gocv.Mat) {

	//width, height := testTool.width, testTool.height
	width, height := 256, 256

	frameA = gocv.NewMatWithSize(height, width, gocv.MatTypeCV8U)
	frameB = gocv.NewMatWithSize(height, width, gocv.MatTypeCV8U)
	frameA2 = gocv.NewMatWithSize(height, width, gocv.MatTypeCV8U)
	frameB2 = gocv.NewMatWithSize(height, width, gocv.MatTypeCV8U)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			frameA.SetUCharAt(y, x, uint8(255))
			frameB.SetUCharAt(y, x, uint8((x)%256))
			frameA2.SetUCharAt(y, x, uint8(255))
			frameB2.SetUCharAt(y, x, uint8((3*x)%256))
		}
	}
	//frameA2.SetUCharAt(12, 12, 0)

	for i := 0; i < 32; i++ {
		for j := 0; j < 32; j++ {
			frameA2.SetUCharAt(width/2-16+i, width/2-16+j, 0)
		}
	}

	return
}

func blockGradient(blockLeftTop Point, blockSide int, gradX, gradY *gocv.Mat) [10]float64 {
	var histogram [10]float64
	avgGradX, avgGradY := 0.0, 0.0
	//fmt.Println("gradY:=====>>>", gradY.Rows(), gradY.Cols())
	//fmt.Println("gradX:=====>>>", gradX.Rows(), gradX.Cols())
	for row := int(blockLeftTop.Y); row < int(blockLeftTop.Y)+blockSide; row++ {
		for col := int(blockLeftTop.X); col < int(blockLeftTop.X)+blockSide; col++ {
			gx := gradX.GetShortAt(row, col)
			gy := gradY.GetShortAt(row, col)
			avgGradX += float64(gx)
			avgGradY += float64(gy)
		}
	}

	blockSize := float64(blockSide * blockSide)
	avgGradX = avgGradX / blockSize
	avgGradY = avgGradY / blockSize
	gradient := [3]float64{avgGradX, avgGradY, float64(0)}

	gradientL2 := norm2Float(gradient[:])
	if gradientL2 == 0.0 {
		return histogram
	}

	gradient[0] = gradient[0] / gradientL2
	gradient[1] = gradient[1] / gradientL2
	gradient[2] = gradient[2] / gradientL2

	// 合并对立方向的梯度值
	for i := 0; i < 10; i++ {
		pi, pi10 := projectGradient(gradient, icosahedronCenterP[i]), projectGradient(gradient, icosahedronCenterP[i+10])
		onePos := math.Abs(pi)
		twoPos := math.Abs(pi10)
		histogram[i] = onePos + twoPos - threshold
		if histogram[i] < 0 {
			histogram[i] = 0.0
		}
		//fmt.Println("(row,col)=>[i:(i,i+10,sum(i)]=>", row, col, i, pi, pi10, onePos, twoPos, project[i])
	}
	pL2 := norm2Float(histogram[:])
	if pL2 == 0.0 {
		return histogram
	}
	for i := 0; i < 10; i++ {
		histogram[i] = histogram[i] / pL2 * gradientL2
		//fmt.Println(project[i])
	}

	return histogram
}

func cellGradient(cellLeftTop, roiCenter Point, cellSide int, sigma float64, gradX, gradY *gocv.Mat) [10]float64 {

	var histogram [10]float64
	blockSide := cellSide / Cell_m
	var centerOfBlock Point
	var blockLeftTop Point
	for blockRow := 0; blockRow < Cell_m; blockRow++ {

		blockLeftTop.Y = cellLeftTop.Y + float64(blockSide*blockRow)
		centerOfBlock.Y = blockLeftTop.Y + float64(blockSide)/2

		for blockCol := 0; blockCol < Cell_m; blockCol++ {
			blockLeftTop.X = cellLeftTop.X + float64(blockSide*blockCol)
			centerOfBlock.X = blockLeftTop.X + float64(blockSide)/2

			hist := blockGradient(blockLeftTop, blockSide, gradX, gradY)
			weight := GaussianKernel2D(centerOfBlock, roiCenter, sigma)

			//fmt.Println("block left-top:", blockLeftTop.String(), "block center:", centerOfBlock.String(), "weight", weight)

			for i := 0; i < 10; i++ {
				histogram[i] += hist[i] * weight
			}
		}
	}
	return histogram
}

//func GaussianKernel2D(a, mua Point, sigma float64) float64 {
//	// 计算高斯函数的分子部分
//	numerator := math.Exp(-((a.X-mua.X)*(a.X-mua.X) + (a.Y-mua.Y)*(a.Y-mua.Y)) / (2 * sigma * sigma))
//	// 计算高斯函数的分母部分，这里省略了，因为通常用于权重计算，常数分母可以不考虑
//	return numerator
//}

func GaussianKernel2D(point, center Point, sigma float64) float64 {
	// 计算两点之间的欧氏距离
	distance := math.Sqrt((point.X-center.X)*(point.X-center.X) + (point.Y-center.Y)*(point.Y-center.Y))
	//fmt.Println(distance)
	// 根据高斯公式计算权重
	x := -((distance * distance) / (2 * sigma * sigma))
	//fmt.Println("x:", x)
	return math.Exp(x)
}

func roiGradient(grayFrame gocv.Mat, roiCenter Point, roiSide int) []float64 {
	gradX := gocv.NewMat()
	gradY := gocv.NewMat()
	defer gradX.Close()
	defer gradY.Close()

	gocv.Sobel(grayFrame, &gradX, gocv.MatTypeCV16S, 1, 0, 3, 1, 0, gocv.BorderDefault)
	gocv.Sobel(grayFrame, &gradY, gocv.MatTypeCV16S, 0, 1, 3, 1, 0, gocv.BorderDefault)
	//fmt.Println("gray frame", grayFrame.Rows(), grayFrame.Cols())
	cellSide := roiSide / Cell_M

	des := make([]float64, 0)
	leftTop := Point{
		X: roiCenter.X - float64(roiSide)/2,
		Y: roiCenter.Y - float64(roiSide)/2,
	}
	//fmt.Println("roi left-top", leftTop.String(), "roi center:", roiCenter.String())
	sigma := 1 //cellSide / SigmaForBaseSize
	var cellLeftTop Point
	for row := 0; row < Cell_M; row++ {
		cellLeftTop.Y = leftTop.Y + float64(row*cellSide)
		for col := 0; col < Cell_M; col++ {
			cellLeftTop.X = leftTop.X + float64(col*cellSide)
			hist := cellGradient(cellLeftTop, roiCenter, cellSide, float64(sigma), &gradX, &gradY)
			des = append(des, hist[:]...)
			//fmt.Println("cell left-top:", cellLeftTop.String())
		}
	}

	l2Norm := norm2Float(des)
	regularizedNorm := l2Norm + 1 // 正则化L2范数加1
	//fmt.Println("roi histogram len:", cellSide, l2Norm, regularizedNorm)
	for i := range des {
		des[i] /= regularizedNorm // 归一化每个元素
	}

	return des
}

func ComputeDesc() {
	frameA, frameA2, frameB, frameB2 := testData()
	defer frameA.Close()
	defer frameA2.Close()
	center := Point{
		X: float64(testTool.cx),
		Y: float64(testTool.cy),
	}
	des := roiGradient(frameA, center, testTool.cs)
	fmt.Println("\nframe A normal=>\n", des)

	des2 := roiGradient(frameA, center, 2*testTool.cs)
	fmt.Println("\nframe A*2=>\n", des2)

	des3 := roiGradient(frameA, center, 4*testTool.cs)
	fmt.Println("\nframe A*4=>\n", des3)

	des4 := roiGradient(frameA2, center, testTool.cs)
	fmt.Println("\nframe A2 normal=>\n", des4)

	des5 := roiGradient(frameA2, center, 2*testTool.cs)
	fmt.Println("\nframe A2*2=>\n", des5)

	des6 := roiGradient(frameA2, center, 4*testTool.cs)
	fmt.Println("\nframe A2*4=>\n", des6)

	des7 := roiGradient(frameB, center, testTool.cs)
	fmt.Println("\nframe B normal=>\n", des7)

	des8 := roiGradient(frameB, center, 2*testTool.cs)
	fmt.Println("\nframe B*2=>\n", des8)

	des9 := roiGradient(frameB, center, 4*testTool.cs)
	fmt.Println("\nframe B*4=>\n", des9)

	des10 := roiGradient(frameB2, center, testTool.cs)
	fmt.Println("\nframe B2 normal=>\n", des10)

	des11 := roiGradient(frameB2, center, 2*testTool.cs)
	fmt.Println("\nframe B2*2=>\n", des11)

	des12 := roiGradient(frameB2, center, 4*testTool.cs)
	fmt.Println("\nframe B2*4=>\n", des12)
}

func getVideoFirstFrame(fileA, fileB string) (gocv.Mat, gocv.Mat) {
	videoA, videoB, err := readFile(fileA, fileB)
	if err != nil {
		panic(fmt.Errorf("failed tor read video file %s", param.rawAFile))
	}
	defer videoA.Close()
	defer videoB.Close()

	frameA, frameB := gocv.NewMat(), gocv.NewMat()
	defer frameA.Close()
	defer frameB.Close()
	if oka, okb := videoA.Read(&frameA), videoB.Read(&frameB); !oka || !okb {
		panic("invalid frame")
	}

	// Convert to grayscale
	grayFrameA, grayFrameB := gocv.NewMat(), gocv.NewMat()
	gocv.CvtColor(frameA, &grayFrameA, gocv.ColorRGBToGray)
	gocv.CvtColor(frameB, &grayFrameB, gocv.ColorRGBToGray)
	return grayFrameA, grayFrameB
}

func calculateL2Distance(histA, histB []float64) float64 {
	sum := 0.0
	for i := 0; i < len(histB); i++ {
		diff := histA[i] - histB[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

func bilinearInterpolate2(x, y float64, w [][]float64, width, height int) float64 {
	// 确保不会越界
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}
	if x >= float64(width) {
		x = float64(width - 1)
	}
	if y >= float64(height) {
		y = float64(height - 1)
	}

	x1 := int(x)
	y1 := int(y)
	x2 := x1
	if x1 < width-1 {
		x2 = x1 + 1
	}
	y2 := y1
	if y1 < height-1 {
		y2 = y1 + 1
	}

	// 插值
	A := w[y1][x1]
	B := w[y1][x2]
	C := w[y2][x1]
	D := w[y2][x2]

	ratioX := x - float64(x1)
	ratioY := y - float64(y1)

	// 计算插值结果
	return A*(1-ratioX)*(1-ratioY) + B*ratioX*(1-ratioY) +
		C*(1-ratioX)*ratioY + D*ratioX*ratioY
}

func ComputeWtl() {
	//frameA, frameA2, _, _ := testData()
	frameA, frameA2 := getVideoFirstFrame("align_A.mp4", "align_B.mp4")
	__saveImg(frameA, "wtl_a.png")
	__saveImg(frameA2, "wtl_b.png")

	defer frameA.Close()
	defer frameA2.Close()
	height := frameA.Rows()
	width := frameA.Cols()
	result := make([][]float64, 0)
	for y := testTool.cs / 2; y <= height-testTool.cs/2; y += StepSize {
		item := make([]float64, 0)
		for x := testTool.cs / 2; x <= width-testTool.cs/2; x += StepSize {

			center := Point{
				X: float64(x),
				Y: float64(y),
			}

			desA := roiGradient(frameA, center, testTool.cs)
			//fmt.Println("frame A normal=>\n", desA)

			desA2 := roiGradient(frameA2, center, testTool.cs)
			//fmt.Println("frame A2 normal=>\n", desA2)

			wtl := calculateL2Distance(desA, desA2)
			//fmt.Println("wtl at:", x, y, wtl)
			item = append(item, wtl)
		}
		result = append(result, item)
	}
	fmt.Println("wtl height:", len(result), "wtl width", len(result[0]))
	a := normalizeImage(result)
	__saveNormalizedData(a, "wtl_wt.png")
}

func wtl(frameA, frameA2 gocv.Mat, roiSize int) [][]float64 {
	height := frameA.Rows()
	width := frameA.Cols()
	wMatrix := make([][]float64, height/StepSize-1)
	for i := range wMatrix {
		wMatrix[i] = make([]float64, width/StepSize-1)
	}

	var row, col = 0, 0
	for y := roiSize / 2; y <= height-roiSize/2; y += StepSize {
		for x := roiSize / 2; x <= width-roiSize/2; x += StepSize {
			center := Point{
				X: float64(x),
				Y: float64(y),
			}
			// 获取当前兴趣区域的描述符
			roiA := roiGradient(frameA, center, roiSize)
			roiA2 := roiGradient(frameA2, center, roiSize)

			// 计算两个描述符之间的L2距离，即Wtl值
			wtl := calculateL2Distance(roiA, roiA2)
			//fmt.Println(roiA, roiA2)
			//fmt.Println("w at center:", wtl, x, y)

			// 将计算得到的Wtl值存储在矩阵中对应的位置
			wMatrix[col][row] = wtl
			row++
		}
		row = 0
		col++
	}
	//fmt.Println(wMatrix)

	return wMatrix
}

func ComputeWt() {
	//frameA, frameA2, _, _ := testData()
	frameA, frameA2 := getVideoFirstFrame("align_a.mp4", "align_b.mp4")
	__saveImg(frameA, "wt_at_raw_a.png")
	__saveImg(frameA2, "wt_at_raw_b.png")
	defer frameA.Close()
	defer frameA2.Close()
	height := frameA.Rows()
	width := frameA.Cols()

	result := make([][]float64, height)
	for i := range result {
		result[i] = make([]float64, width)
	}

	for l := 0; l < 3; l++ {
		interpolatedW := make([][]float64, height)
		for i := range interpolatedW {
			interpolatedW[i] = make([]float64, width)
		}
		times := 1 << l
		roiSize := BaseSizeOfPixel * times

		wMatrix := wtl(frameA, frameA2, roiSize)

		// 应用双线性插值
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				w, h := len(wMatrix[0]), len(wMatrix)
				wtlxy := bilinearInterpolate2(float64(x)/float64(StepSize), float64(y)/float64(StepSize), wMatrix, w, h)
				interpolatedW[y][x] = wtlxy
				result[y][x] = result[y][x] + wtlxy*float64(times)

				//if result[y][x] > 1 {
				//	fmt.Println("good result", result[y][x])
				//}

				//if interpolatedW[y][x] != result[y][x] {
				//	fmt.Println(result[y][x], interpolatedW[y][x])
				//}
			}
		}
		a := normalizeImage(interpolatedW)
		saveJson(fmt.Sprintf("wt_at_level_%d.txt", times), a)
		__saveNormalizedData(a, fmt.Sprintf("wt_at_level_%d.png", times))
	}
	a := normalizeImage(result)
	saveJson("wt_at_level_0.txt", a)
	__saveNormalizedData(a, "wt_at_level_0.png")
}

func computeG(grayFrameB gocv.Mat) [][]float64 {

	floatInput := gocv.NewMat()
	grayFrameB.ConvertToWithParams(&floatInput, gocv.MatTypeCV32F, 1.0/255, 0) // 注意添加归一化

	fmt.Println("gray frame b:", grayFrameB.Type(), "float input :", floatInput.Type())

	// 初始化Sobel梯度矩阵
	gradX := gocv.NewMat()
	gradY := gocv.NewMat()
	gocv.Sobel(floatInput, &gradX, gocv.MatTypeCV32F, 1, 0, 3, 1, 0, gocv.BorderDefault)
	gocv.Sobel(floatInput, &gradY, gocv.MatTypeCV32F, 0, 1, 3, 1, 0, gocv.BorderDefault)
	floatInput.Close() // 释放转换后的Mat

	// 计算梯度幅值
	gradientMagnitude := make([][]float64, grayFrameB.Rows())
	for y := 0; y < grayFrameB.Rows(); y++ {
		gradientMagnitude[y] = make([]float64, grayFrameB.Cols())
		for x := 0; x < grayFrameB.Cols(); x++ {
			gx := gradX.GetFloatAt(y, x)
			gy := gradY.GetFloatAt(y, x)
			//if gx != 0 || gy != 0 {
			//	fmt.Println("computeG:", gx, gy)
			//}

			// 计算梯度的大小并应用alpha值
			g := math.Sqrt(float64(gx*gx + gy*gy))
			g = math.Min(1, testTool.alpha*g)

			gradientMagnitude[y][x] = g
		}
	}

	// 清理资源
	gradX.Close()
	gradY.Close()

	return gradientMagnitude
}

func ComputeG() {
	frameA, frameB := getVideoFirstFrame("align_A.mp4", "align_B.mp4")
	defer frameA.Close()
	defer frameB.Close()
	gradB := computeG(frameB)

	__saveNormalizedData(gradB, "wt_grad_b.png")
	//a := normalizeImage(gradB)
	//__saveNormalizedData(a, "wt_grad_b_2.png")
}

func ComputeG2() {
	img := gocv.IMRead("ai_source.jpg", gocv.IMReadColor)
	if img.Empty() {
		fmt.Println("无法读取图片")
		return
	}

	grayFrameA := gocv.NewMat()
	gocv.CvtColor(img, &grayFrameA, gocv.ColorRGBToGray)
	defer img.Close()
	gradB := computeG(grayFrameA)

	__saveNormalizedData(gradB, "ai_dest.png")
	//a := normalizeImage(gradB)
	//__saveNormalizedData(a, "wt_grad_b_2.png")
}

func convertMatToIntSlice(mat gocv.Mat) []int {
	height, width := mat.Rows(), mat.Cols()
	slice := make([]int, height*width)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			pixelValue := mat.GetUCharAt(y, x) // Assuming mat is of type CV_8U
			slice[y*width+x] = int(pixelValue)
		}
	}

	return slice
}

func calculatePercentile(slice []int, percentile float64) int {
	sort.Ints(slice)
	index := int((percentile / 100) * float64(len(slice)-1))
	return slice[index]
}

func adjustContrast(frameA gocv.Mat, betaLow, betaHigh float64) [][]float64 {
	slice := convertMatToIntSlice(frameA)
	Ilow := calculatePercentile(slice, 1)
	Ihigh := calculatePercentile(slice, 99)
	adjusted := make([][]float64, frameA.Rows())
	factor := (betaHigh - betaLow) / float64(Ihigh-Ilow)
	//adjusted := gocv.NewMatWithSize(frameA.Rows(), frameA.Cols(), gocv.MatTypeCV8U)

	for y := 0; y < frameA.Rows(); y++ {
		adjusted[y] = make([]float64, frameA.Cols())
		for x := 0; x < frameA.Cols(); x++ {
			originalValue := int(frameA.GetUCharAt(y, x))
			scaledValue := float64(originalValue-Ilow)*factor + betaLow
			clippedValue := math.Min(math.Max(scaledValue, betaLow), betaHigh)
			//adjusted.SetUCharAt(y, x, uint8(clippedValue*255))
			adjusted[y][x] = clippedValue
		}
	}

	return adjusted
}

func AdjustContrast() {
	frameA, frameB := getVideoFirstFrame("align_A.mp4", "align_B.mp4")
	defer frameA.Close()
	defer frameB.Close()
	adMat := adjustContrast(frameA, testTool.betaLow, testTool.betaHigh)
	__saveNormalizedData(adMat, "wt_adjust_a.png")
	//__saveImg(adMat, "wt_adjust_a.png")
}

func linearInterpolation(colorA, colorB color.RGBA, factor float64) color.RGBA {
	return color.RGBA{
		R: uint8(float64(colorA.R)*(1-factor) + float64(colorB.R)*factor),
		G: uint8(float64(colorA.G)*(1-factor) + float64(colorB.G)*factor),
		B: uint8(float64(colorA.B)*(1-factor) + float64(colorB.B)*factor),
		A: 255, // Alpha值保持不变，总是不透明
	}
}

func fc(score float64) color.RGBA {
	// 确保分数在0和1之间
	//score = math.Max(0, math.Min(score, 1))
	if score > 1 || score < 0 {
		panic(fmt.Sprintf("invalid score:%.2f", score))
	}

	// 定义颜色值
	lowColor := color.RGBA{R: 255, G: 253, B: 175, A: 255}
	highColor := color.RGBA{R: 255, G: 0, B: 0, A: 255}

	// 根据分数进行颜色插值
	interpolatedColor := linearInterpolation(lowColor, highColor, score)

	return interpolatedColor
}

func ComputeFC() {
	frameA, frameB := getVideoFirstFrame("align_A.mp4", "align_B.mp4")
	defer frameA.Close()
	defer frameB.Close()

	width := frameA.Cols()
	height := frameA.Rows()
	var wtVal [][]float64
	_ = readJson("wt_at_level_0.txt", &wtVal)

	img := image.NewRGBA(image.Rect(0, 0, width, height))
	//aColor := color.RGBA{R: grayValue, G: grayValue, B: grayValue, A: 255}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			aColor := fc(wtVal[y][x])
			//if wtVal[y][x] > 0.95 {
			//	fmt.Println("score======>>>>", wtVal[y][x], y, x)
			//}
			img.Set(x, y, aColor)
		}
	}
	file, _ := os.Create("overlay_fc.png")
	defer file.Close()
	png.Encode(file, img)
}

// Note: You will need to implement the calculatePercentile function that
// calculates the given percentile of a slice of float64 values.

func ComputeOverlay() {
	var wtVal [][]float64
	_ = readJson("wt_at_level_0.txt", &wtVal)

	frameA, frameB := getVideoFirstFrame("align_A.mp4", "align_B.mp4")
	defer frameA.Close()
	defer frameB.Close()

	gradientMagnitude := computeG(frameB)

	img := overlay(frameA, wtVal, gradientMagnitude)

	file, _ := os.Create("overlay_result.png")
	defer file.Close()
	png.Encode(file, img)
}

func wt(frameA, frameB gocv.Mat) [][]float64 {
	height := frameA.Rows()
	width := frameA.Cols()

	result := make([][]float64, height)
	for i := range result {
		result[i] = make([]float64, width)
	}

	for l := 0; l < 3; l++ {
		interpolatedW := make([][]float64, height)
		for i := range interpolatedW {
			interpolatedW[i] = make([]float64, width)
		}
		times := 1 << l
		roiSize := BaseSizeOfPixel * times

		wMatrix := wtl(frameA, frameB, roiSize)

		// 应用双线性插值
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				w, h := len(wMatrix[0]), len(wMatrix)
				wtlxy := bilinearInterpolate2(float64(x)/float64(StepSize), float64(y)/float64(StepSize), wMatrix, w, h)
				interpolatedW[y][x] = wtlxy
				result[y][x] = result[y][x] + wtlxy*float64(times)
			}
		}
	}
	return normalizeImage(result)
}

func overlay(frameA gocv.Mat, wtVal, gradientMagnitude [][]float64) image.Image {
	width := frameA.Cols()
	height := frameA.Rows()
	adjustedFrame := adjustContrast(frameA, testTool.betaLow, testTool.betaHigh)

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

func ComputeVideoDiff() {
	videoA, videoB, err := readFile(param.alignedAFile, param.alignedBFile)
	if err != nil {
		panic("failed tor read video file")
	}
	defer videoA.Close()
	defer videoB.Close()

	width := videoA.Get(gocv.VideoCaptureFrameWidth)
	height := videoA.Get(gocv.VideoCaptureFrameHeight)
	fps := videoA.Get(gocv.VideoCaptureFPS)

	videoWriter, _ := gocv.VideoWriterFile(
		"overlay_result.mp4", // 输出视频文件
		"mp4v",               // 编码格式
		fps,                  // FPS
		int(width),           // 视频宽度
		int(height),          // 视频高度
		true)                 // 是否彩色
	defer videoWriter.Close()

	var idx = 0
	for {
		var frameA = gocv.NewMat()
		if ok := videoA.Read(&frameA); !ok || frameA.Empty() {
			frameA.Close()
			break
		}

		var frameB = gocv.NewMat()
		if ok := videoB.Read(&frameB); !ok || frameB.Empty() {
			videoB.Close()
			break
		}
		grayFrameA, grayFrameB := gocv.NewMat(), gocv.NewMat()
		gocv.CvtColor(frameA, &grayFrameA, gocv.ColorRGBToGray)
		gocv.CvtColor(frameB, &grayFrameB, gocv.ColorRGBToGray)
		frameA.Close()
		frameB.Close()

		wtVal := wt(grayFrameA, grayFrameB)
		gradientMagnitude := computeG(grayFrameB)
		img := overlay(grayFrameA, wtVal, gradientMagnitude)

		file, _ := os.Create(fmt.Sprintf("tmp/overlay/overlay_result_%d.png", idx))
		_ = png.Encode(file, img)
		_ = file.Close()

		mat, err := gocv.ImageToMatRGB(img)
		if err != nil {
			panic(err)
		}

		_ = videoWriter.Write(mat)
		grayFrameA.Close()
		grayFrameB.Close()
		idx++
		fmt.Println("finish frame:", idx)
	}
}

func ComputeDiffImg() {
	imgA := gocv.IMRead("wtl_c.jpeg", gocv.IMReadGrayScale)
	imgB := gocv.IMRead("wtl_d.jpeg", gocv.IMReadGrayScale)
	//__saveImg(img, "wtl_e.jpeg")

	wtVal := wt(imgA, imgB)
	gradientMagnitude := computeG(imgB)
	img := overlay(imgA, wtVal, gradientMagnitude)

	file, _ := os.Create("wtl_c_d_overlay.png")
	_ = png.Encode(file, img)
	_ = file.Close()
}

func AITest() {
	// 加载原始图片
	src := gocv.IMRead("ai_source.jpg", gocv.IMReadColor)
	defer src.Close()

	// 转换为灰度图像
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(src, &gray, gocv.ColorRGBToGray)

	// 应用高斯模糊减少噪点
	gocv.GaussianBlur(gray, &gray, image.Point{X: 9, Y: 9}, 0, 0, gocv.BorderDefault)

	// 应用 Canny 边缘检测，调整阈值
	edges := gocv.NewMat()
	defer edges.Close()
	gocv.Canny(gray, &edges, 100, 200)

	// 使用形态学操作去除小噪点
	kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Point{X: 5, Y: 5})
	gocv.Dilate(edges, &edges, kernel)

	// 查找轮廓
	contours := gocv.FindContours(edges, gocv.RetrievalExternal, gocv.ChainApproxSimple)

	// 创建一个空白的黑色图像
	result := gocv.NewMatWithSize(src.Rows(), src.Cols(), gocv.MatTypeCV8U)
	defer result.Close()
	result.SetTo(gocv.NewScalar(0, 0, 0, 0))

	// 绘制白色轮廓，过滤小轮廓
	white := color.RGBA{255, 255, 255, 255}
	for i := 0; i < contours.Size(); i++ {
		if gocv.ContourArea(contours.At(i)) > 100 { // 只绘制较大的轮廓
			gocv.DrawContours(&result, contours, i, white, -1) // -1 填充轮廓
		}
	}

	// 保存或显示结果
	window := gocv.NewWindow("Contours")
	defer window.Close()

	for {
		window.IMShow(result)
		if window.WaitKey(1) >= 0 {
			break
		}
	}
}
