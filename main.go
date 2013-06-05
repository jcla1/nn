package main

import (
	"fmt"
	"github.com/jcla1/matrix"
	"math"
)

func main() {
	input := matrix.FromMatlab("[1;1]")
	thetas := []*matrix.Matrix{matrix.FromMatlab("[-30 20 20]")}

	fmt.Println(hypothesis(thetas, input))
}

func hypothesis(thetas []*matrix.Matrix, input *matrix.Matrix) *matrix.Matrix {
	// Describes the current working values (a_1, a_2, ...)
	curValues := input

	// Is simply a 1 in a 1x1 matrix to b
	// inserted into a vector as the bias unit
	biasValueMatrix := matrix.Ones(1, 1)

	for _, theta := range thetas {
		// Insert the bias unit, multiply with theta and apply the sigmoid function
		curValues = theta.Mul(curValues.InsertRows(biasValueMatrix, 0)).Apply(sigmoidMatrix)
	}

	return curValues
}

func sigmoidMatrix(index int, value float64) float64 {
	return sigmoid(value)
}

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Pow(math.E, -z))
}
