package main

import (
	"fmt"
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
	op     int
	width  int
	height int
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

func gradient(grayFrame, prevFrame gocv.Mat) {
	gradX := gocv.NewMat()
	gradY := gocv.NewMat()
	gradT := gocv.NewMat()

	gocv.Sobel(grayFrame, &gradX, gocv.MatTypeCV16S, 1, 0, 3, 1, 0, gocv.BorderDefault)
	gocv.Sobel(grayFrame, &gradY, gocv.MatTypeCV16S, 0, 1, 3, 1, 0, gocv.BorderDefault)
	gocv.AbsDiff(grayFrame, prevFrame, &gradT)

	histogram := quantizeGradients(&gradX, &gradY, &gradT)

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
	gradient(frameB, frameB2)
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

	width, height := testTool.width, testTool.height

	frameA = gocv.NewMatWithSize(height, width, gocv.MatTypeCV8U)
	frameB = gocv.NewMatWithSize(height, width, gocv.MatTypeCV8U)
	frameA2 = gocv.NewMatWithSize(height, width, gocv.MatTypeCV8U)
	frameB2 = gocv.NewMatWithSize(height, width, gocv.MatTypeCV8U)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			frameA.SetUCharAt(y, x, uint8((x)%256))
			frameB.SetUCharAt(y, x, uint8((x)%256))
			frameA2.SetUCharAt(y, x, uint8((2*x)%256))
			frameB2.SetUCharAt(y, x, uint8((2*x)%256))
		}
	}
	return
}
