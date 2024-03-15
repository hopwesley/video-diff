package main

import (
	"fmt"
	"github.com/spf13/cobra"
	"math"
)

type startParam struct {
	version      bool
	rawAFile     string
	rawBFile     string
	alignedAFile string
	alignedBFile string
}

const (
	Version = "0.1.1"
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

func init() {
	flags := rootCmd.Flags()
	flags.BoolVarP(&param.version, "version",
		"v", false, "golf -v")

	flags.StringVarP(&param.alignedAFile, "source",
		"a", "A.mp4", "golf -s A.mp4")

	flags.StringVarP(&param.alignedBFile, "dest",
		"b", "B.mp4", "golf -d B.mp4")

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
}
