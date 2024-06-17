package main

import (
	"fmt"
	"github.com/spf13/cobra"
	"gocv.io/x/gocv"
	"math"
	"sort"
)

const (
	//PercentForMaxDepthToTimeAlign = 0.5 //20%of all frames to find the time start
	PrefixForAlignedFile = "align_"
	MaxPairToMatch       = 3
	threshold            = 1.29107 // 根据论文描述的阈值

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
		"a", "A.mp4", "golf -a A.mp4")

	flags.StringVarP(&param.rawBFile, "dest",
		"b", "B.mp4", "golf -b B.mp4")
	flags.IntVarP(&param.alignGap, "gap", "g", 60, "-g gap about similarity center")
}
func saveAlign(idx int, match [2]int, videoA, videoB *gocv.VideoCapture) {
	idxA, idxB := match[0], match[1]
	if idxB < 0 || idxA < 0 {
		panic("find time start frame failed")
	}
	fmt.Println("time align =>", idx, idxA, idxB)
	var startA, startB = 0, 0
	if idxA > idxB {
		startA = idxA - idxB
		startB = 0
	} else {
		startB = idxB - idxA
		startA = 0
	}
	fmt.Println("time align =>", idx, startA, startB)

	saveVideoFromFrame(videoA, startA, fmt.Sprintf("q_%d_", idx)+PrefixForAlignedFile+param.rawAFile)
	saveVideoFromFrame(videoB, startB, fmt.Sprintf("q_%d_", idx)+PrefixForAlignedFile+param.rawBFile)

	saveVideoFromFrame(videoA, idxA, fmt.Sprintf("s_q_%d_", idx)+PrefixForAlignedFile+param.rawAFile)
	saveVideoFromFrame(videoB, idxB, fmt.Sprintf("s_q_%d_", idx)+PrefixForAlignedFile+param.rawBFile)
}

func alignRun2(_ *cobra.Command, _ []string) {
	videoA, videoB, err := readFile(param.rawAFile, param.rawBFile)
	if err != nil {
		panic(fmt.Errorf("failed tor read video file %s", param.rawAFile))
	}
	defer videoA.Close()
	defer videoB.Close()

	aHisGram, _ := parseHistogram(videoA)
	bHisGram, _ := parseHistogram(videoB)

	// 应用阈值处理

	aHisGramFloat := distributeGradientMagnitude(aHisGram, threshold)
	bHisGramFloat := distributeGradientMagnitude(bHisGram, threshold)

	//idxA, idxB := findTimeStartOfFrame(aHisGramFloat, bHisGramFloat)
	aligns := findTopThreeMatches(aHisGramFloat, bHisGramFloat)
	for i, align := range aligns {
		fmt.Println("queue:", i)
		saveAlign(i, align, videoA, videoB)
	}
	//aHisGramFloat = aHisGramFloat[startA:]
	//bHisGramFloat = bHisGramFloat[startB:]
}

func alignRun(_ *cobra.Command, _ []string) {
	videoA, videoB, err := readFile(param.rawAFile, param.rawBFile)
	if err != nil {
		panic(fmt.Errorf("failed tor read video file %s", param.rawAFile))
	}
	defer videoA.Close()
	defer videoB.Close()

	aHisGram, _ := parseHistogram2(videoA)
	bHisGram, _ := parseHistogram2(videoB)
	ncc := nccOfAllFrame(aHisGram, bHisGram)

	startA, startB := findMaxNCCSequence(ncc, testTool.window)
	saveVideoFromFrame(videoA, startA, "align_"+param.rawAFile)
	saveVideoFromFrame(videoB, startB, "align_"+param.rawBFile)
}

type Match struct {
	IndexA int     // 视频A中帧的索引
	IndexB int     // 视频B中帧的索引
	Score  float64 // 相似度分数
}

func isFrameAlreadySelected(matches [3][2]int, match Match) bool {
	for _, m := range matches {
		if m[0] == match.IndexA || m[1] == match.IndexB {
			return true
		}
	}
	return false
}

func findTopThreeMatches(aHisGramFloat, bHisGramFloat [][]float64) (matches [3][2]int) {
	var allMatches []Match // Match 是一个结构体，包含两个视频中帧的索引和它们之间的相似度分数
	for i, histA := range aHisGramFloat {
		for j, histB := range bHisGramFloat {
			score := calculateNCC(histA, histB) // 计算帧对 (i, j) 之间的相似度分数
			allMatches = append(allMatches, Match{i, j, score})
		}
	}

	// 根据相似度分数对所有匹配进行排序，这里假设有一个自定义的比较函数
	sort.Slice(allMatches, func(i, j int) bool {
		return allMatches[i].Score > allMatches[j].Score
	})

	// 选取前三个最匹配的帧对，同时确保每个视频中的每帧只被选取一次
	selected := 0
	for _, match := range allMatches {
		if selected == MaxPairToMatch {
			break
		}
		if !isFrameAlreadySelected(matches, match) {
			matches[selected] = [2]int{match.IndexA, match.IndexB}
			selected++
		}
	}
	return matches
}

func nccOfAllFrame(aHisGramFloat, bHisGramFloat [][]float64) [][]float64 {

	videoALength := len(aHisGramFloat) // Video A frame count
	videoBLength := len(bHisGramFloat) // Video B frame count

	// Initialize a 2D array to store the NCC values
	nccValues := make([][]float64, videoALength)
	for i := range nccValues {
		nccValues[i] = make([]float64, videoBLength)
	}
	// Iterate over all frame pairs of Video A and Video B, calculate their NCC values
	for i, histogramA := range aHisGramFloat {
		for j, histogramB := range bHisGramFloat {
			nccValues[i][j] = calculateNCC(histogramA, histogramB)
		}
	}
	return nccValues // These are the indices of the frames that best align in time
}

func findMaxNCCSequence(nccValues [][]float64, sequenceLength int) (int, int) {
	maxSum := -1.0       // 假设NCC值范围是-1到1，开始时设置为最小可能的和
	maxI, maxJ := -1, -1 // 用于存储最大和对应的起始帧索引

	for i := 0; i <= len(nccValues)-sequenceLength; i++ {
		for j := 0; j <= len(nccValues[0])-sequenceLength; j++ {
			sum := 0.0
			for k := 0; k < sequenceLength; k++ {
				sum += nccValues[i+k][j+k] // 计算连续sequenceLength帧的NCC值之和
			}

			if sum > maxSum {
				maxSum = sum
				maxI, maxJ = i, j
			}
		}
	}
	return maxI, maxJ // 返回连续sequenceLength帧NCC值之和最大的起始帧索引
}

func computeFrameVector(quantizedGradients [][][]float64) []float64 {
	frameVector := make([]float64, 10) // 一个帧的10维向量 q_t^A

	// 遍历每个像素的量化梯度向量
	for _, row := range quantizedGradients {
		for _, pixelVector := range row {
			for i, value := range pixelVector {
				//fmt.Println(x, y, value, frameVector[i])
				frameVector[i] += value // 对每一维度进行累加
			}
		}
	}

	// 归一化 frameVector
	//norm := norm2Float(frameVector)
	//if norm > 0 {
	//	for i := range frameVector {
	//		frameVector[i] /= norm // 对每一维度的值进行归一化
	//	}
	//}

	return frameVector
}

func parseHistogram2(video *gocv.VideoCapture) ([][]float64, error) {
	// 初始化前一帧变量
	var prevFrame gocv.Mat
	firstFrame := true
	var idx = 0
	histograms := make([][]float64, 0)
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
		gocv.CvtColor(frame, &grayFrame, gocv.ColorRGBToGray)
		frame.Close()
		if firstFrame {
			fmt.Println("[parseHistogram] read first frame from video", idx)
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
		histogram := quantizeGradients2(&gradX, &gradY, &gradT)
		sumHistogram := computeFrameVector(histogram)
		fmt.Println("[parseHistogram] parse histogram for frame:", idx, sumHistogram)

		histograms = append(histograms, sumHistogram) // 将当前帧的直方图添加到数组中

		gradX.Close()
		gradY.Close()
		gradT.Close()
		grayFrame.Close()
	}

	// Release the last previous frame
	if !prevFrame.Empty() {
		prevFrame.Close()
	}
	return histograms, nil
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
		gocv.CvtColor(frame, &grayFrame, gocv.ColorRGBToGray)
		frame.Close()
		if firstFrame {
			fmt.Println("[parseHistogram] read first frame from video", idx)
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
		histogram := quantizeGradients(&gradX, &gradY, &gradT)
		fmt.Println("[parseHistogram] parse histogram for frame:", idx)

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
