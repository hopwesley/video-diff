package main

import (
	"github.com/spf13/cobra"
	"gocv.io/x/gocv"
)

var testCmd = &cobra.Command{
	Use: "test",

	Short: "test",

	Long: `usage description::TODO::`,

	Run: testRun,
}

type testParam struct {
	op   int
	size int
}

var testTool = &testParam{}

func init() {
	flags := testCmd.Flags()

	flags.IntVarP(&testTool.op, "operation",
		"o", 1, "golf test -o 1")

	flags.IntVarP(&testTool.size, "size",
		"s", BaseSizeOfPixel, "golf test -s 128")
}
func testRun(_ *cobra.Command, _ []string) {
	switch testTool.op {
	case 1:
		createTestImg()
		return
	case 2:
		ComputeGradient()
		return
	}
}
func ComputeGradient() {

}
func createTestImg() {
	width, height := testTool.size, testTool.size
	frameA := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8U)
	frameB := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8U)

	defer frameA.Close()
	defer frameB.Close()

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			frameA.SetUCharAt(y, x, uint8(x%256))
			frameB.SetUCharAt(y, x, uint8((256-x)%256))
		}
	}

	__saveImg(frameA, "test/t_a.png")
	__saveImg(frameB, "test/t_b.png")
}
