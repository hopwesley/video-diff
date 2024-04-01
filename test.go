package main

import (
	"fmt"
	"github.com/spf13/cobra"
	"gocv.io/x/gocv"
	"math"
)

var testCmd = &cobra.Command{
	Use: "test",

	Short: "test",

	Long: `usage description::TODO::`,

	Run: testRun,
}

type testParam struct {
	op     int
	width  int
	height int
	window int
	cx     int
	cy     int
	cs     int
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
		"w", 3, "golf test -w 3")

	flags.StringVarP(&param.rawAFile, "source",
		"a", "A.mp4", "golf -a A.mp4")

	flags.StringVarP(&param.rawBFile, "dest",
		"b", "B.mp4", "golf -b B.mp4")

	flags.IntVarP(&testTool.cx, "centerX",
		"c", 32, "golf test -c 64")
	flags.IntVarP(&testTool.cy, "centerY",
		"d", 32, "golf test -d 64")
	flags.IntVarP(&testTool.cs, "size",
		"s", 16, "golf test -s 32")
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
		ComputeW()
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
	aHisGram, _ := readJson("a_histogram.txt")
	bHisGram, _ := readJson("b_histogram.txt")
	ncc := nccOfAllFrame(aHisGram, bHisGram)
	saveJson("ncc.txt", ncc)
	startA, startB := findMaxNCCSequence(ncc, testTool.window)
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
	width, height := 128, 128

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
	frameA2.SetUCharAt(12, 12, 0)

	//for i := 0; i < 32; i++ {
	//	for j := 0; j < 32; j++ {
	//		frameA2.SetUCharAt(width/2-16+i, width/2-16+j, 0)
	//	}
	//}

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

func GaussianKernel2D(a, mua Point, sigma float64) float64 {
	// 计算高斯函数的分子部分
	numerator := math.Exp(-((a.X-mua.X)*(a.X-mua.X) + (a.Y-mua.Y)*(a.Y-mua.Y)) / (2 * sigma * sigma))
	// 计算高斯函数的分母部分，这里省略了，因为通常用于权重计算，常数分母可以不考虑
	return numerator
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
	gocv.CvtColor(frameA, &grayFrameA, gocv.ColorBGRToGray)
	gocv.CvtColor(frameB, &grayFrameB, gocv.ColorBGRToGray)
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
	frameA, frameA2, _, _ := testData()
	__saveImg(frameA, "wtl_a.png")
	__saveImg(frameA2, "wtl_b.png")
	defer frameA.Close()
	defer frameA2.Close()
	height := frameA.Rows()
	width := frameA.Cols()
	center := Point{
		X: float64(testTool.cx),
		Y: float64(testTool.cy),
	}
	for y := testTool.cs / 2; y <= height-testTool.cs/2; y += StepSize {
		for x := testTool.cs / 2; x <= width-testTool.cs/2; x += StepSize {
			desA := roiGradient(frameA, center, testTool.cs)
			fmt.Println("\nframe A normal=>\n", desA)

			desA2 := roiGradient(frameA2, center, testTool.cs)
			fmt.Println("\nframe A2 normal=>\n", desA2)

			wtl := calculateL2Distance(desA, desA2)
			fmt.Println("\n wtl at:", x, y, wtl)
		}
	}
	//desA := roiGradient(frameA, center, testTool.cs)
	//fmt.Println("\nframe A normal=>\n", desA)
	//
	//desA2 := roiGradient(frameA2, center, testTool.cs)
	//fmt.Println("\nframe A2 normal=>\n", desA2)
	//
	//wtl := calculateL2Distance(desA, desA2)
	//fmt.Println("\n w at l=1", wtl)
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

func ComputeW() {
	//frameA, frameA2, _, _ := testData()
	frameA, frameA2 := getVideoFirstFrame("align_A.mp4", "align_B.mp4")
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
				//if interpolatedW[y][x] != result[y][x] {
				//	fmt.Println(result[y][x], interpolatedW[y][x])
				//}
			}
		}

		normalizeAndConvertToImage(interpolatedW, fmt.Sprintf("wt_at_level_%d.png", times))
	}

	normalizeAndConvertToImage(result, "wt_at_level_0.png")
}
