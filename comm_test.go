package main

import (
	"fmt"
	"testing"
)

func TestThreshold(t *testing.T) {
	for i, _ := range icosahedronCenterP {
		var projection = projectGradient(icosahedronCenterP[i], icosahedronCenterP[(i+1)%len(icosahedronCenterP)])
		fmt.Println("projection=>", projection)
	}
}
