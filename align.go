package main

import (
	"fmt"
	"github.com/spf13/cobra"
	"gocv.io/x/gocv"
	"math"
)

const (
	PercentForMaxDepthToTimeAlign = 0.3 //20%of all frames to find the time start
	PrefixForAlignedFile          = "align_"
)

var alignCmd = &cobra.Command{
	Use: "align",

	Short: "align",

	Long: `usage description::TODO::`,

	Run: alignRun,
}

func init() {
	flags := alignCmd.Flags()

	flags.StringVarP(&param.rawAFile, "source",
		"a", "A.mp4", "golf -s A.mp4")

	flags.StringVarP(&param.rawBFile, "dest",
		"b", "B.mp4", "golf -d B.mp4")
}

func alignRun(_ *cobra.Command, _ []string) {
	videoA, videoB, err := readFile(param.rawAFile, param.rawBFile)
	if err != nil {
		panic(fmt.Errorf("failed tor read video file %s", param.rawAFile))
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

	saveVideoFromFrame(videoA, startA, PrefixForAlignedFile+param.rawAFile)
	saveVideoFromFrame(videoB, startB, PrefixForAlignedFile+param.rawBFile)
	//aHisGramFloat = aHisGramFloat[startA:]
	//bHisGramFloat = bHisGramFloat[startB:]
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

func parseHistogram(video *gocv.VideoCapture) ([][]int, error) {

	// 初始化前一帧变量
	var prevFrame gocv.Mat
	firstFrame := true

	var histograms [][]int // 用于存储每一帧的直方图
	var idx = 0
	for {
		var frame = gocv.NewMat()
		if ok := video.Read(&frame); !ok || frame.Empty() {
			fmt.Println("[parseHistogram] read frame from video finished", idx)
			frame.Close()
			break
		}
		idx++
		// Convert to grayscale
		var grayFrame = gocv.NewMat()
		gocv.CvtColor(frame, &grayFrame, gocv.ColorBGRToGray)
		frame.Close()
		if firstFrame {
			fmt.Println("[parseHistogram] read first frame from video", idx)
			firstFrame = false
			prevFrame = grayFrame.Clone()
			continue
		}
		fmt.Println("[parseHistogram] read new frame from video", idx)

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

// 分配梯度幅度和应用阈值
func distributeGradientMagnitude(hists [][]int, threshold float64) [][]float64 {
	processedHists := make([][]float64, len(hists))
	fmt.Println("apply threshold to q^")
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
