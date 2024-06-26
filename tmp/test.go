package main

import (
	"fmt"
	"math"
)

// 定义 Histogram 结构体
type Histogram [10]float64

// 加法运算
func (h *Histogram) Add(hg Histogram) {
	for i := 0; i < 10; i++ {
		h[i] += hg[i]
	}
}

// 点积计算
func (h *Histogram) dotProduct(h2 Histogram) float64 {
	var sum float64
	for i := 0; i < 10; i++ {
		sum += h[i] * h2[i]
	}
	return sum
}

// L2范数计算
func (h *Histogram) l2Norm() float64 {
	var sum float64
	for i := 0; i < 10; i++ {
		sum += h[i] * h[i]
	}
	return math.Sqrt(sum)
}

// 计算Histogram的L2范数
func (h *Histogram) length() float64 {
	var sum float64
	for i := 0; i < 10; i++ {
		sum += h[i] * h[i]
	}
	return sum
}

// 计算归一化交叉相关性
func normalizedCrossCorrelation(AQ, BQ []Histogram) ([]float64, float64) {
	maxCorr := -1.0
	lenA := len(AQ)
	lenB := len(BQ)
	correlations := make([]float64, lenA+lenB-1)

	for offset := -lenB + 1; offset < lenA; offset++ {
		sumDotProduct := 0.0
		sumNormA := 0.0
		sumNormB := 0.0
		count := 0

		for i := 0; i < lenA; i++ {
			j := i - offset
			if j >= 0 && j < lenB {
				dotProduct := AQ[i].dotProduct(BQ[j])
				normA := AQ[i].l2Norm()
				normB := BQ[j].l2Norm()

				sumDotProduct += dotProduct
				sumNormA += normA
				sumNormB += normB
				count++
			}
		}

		if count > 0 {
			normFactor := math.Sqrt(sumNormA * sumNormB)
			if normFactor > 0 {
				corr := sumDotProduct / normFactor
				correlations[offset+lenB-1] = corr
				if corr > maxCorr {
					maxCorr = corr
				}
			}
		}
	}

	return correlations, maxCorr
}

func main() {
	// 示例 AQ 和 BQ 的初始化
	var AQ = []Histogram{
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		{2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
		{3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		{4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
		{5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
		{6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
	}

	var BQ = []Histogram{
		{3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		{2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
		{4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
		{5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
		{6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
		{7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
	}
	a, b := normalizedCrossCorrelation(AQ, BQ)
	fmt.Printf("%v:%v\n", a, b)
}
