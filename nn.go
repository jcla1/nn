package nn

import (
	"fmt"
	"github.com/jcla1/matrix"
	"math"
)

var (
	_ = fmt.Println
)

type TrainingExample struct {
	Input, ExpectedOutput *matrix.Matrix
}

type Parameters []*matrix.Matrix
type Deltas []*matrix.Matrix

func CostFunction(data []TrainingExample, thetas Parameters, lambda float64) float64 {
	cost := float64(0)
	var estimation []float64
	var expected_output []float64

	// Cost

	for _, datum := range data {
		estimation = Hypothesis(thetas, datum).Values()
		expected_output = datum.ExpectedOutput.Values()

		for k, y := range expected_output {
			// heart of the cost function
			cost += y*math.Log(estimation[k]) + (1-y)*math.Log(1-estimation[k])
		}
	}

	// Regularization
	regularizationCost := float64(0)

	for _, theta := range thetas {
		for i, param := range theta.Values() {

			// ignore theta0
			if i%theta.Columns() == 0 {
				continue
			}

			regularizationCost += param * param
		}
	}

	return cost/float64(len(data)) + (lambda/(2*float64(len(data))))*regularizationCost
}

func Hypothesis(thetas Parameters, trainingEx TrainingExample) *matrix.Matrix {
	// Describes the current working values (a_1, a_2, ...)
	curValues := trainingEx.Input

	// Is simply a 1 in a 1x1 matrix to b
	// inserted into a vector as the bias unit
	biasValueMatrix := matrix.Ones(1, 1)

	for _, theta := range thetas {
		// Insert the bias unit, multiply with theta and apply the sigmoid function
		curValues = theta.Mul(curValues.InsertRows(biasValueMatrix, 0)).Apply(sigmoidMatrix)
	}

	return curValues
}

func DeltaTerms(thetas Parameters, trainingEx TrainingExample) Deltas {
	deltas := make(Deltas, len(thetas))

	biasValueMatrix := matrix.Ones(1, 1)

	deltas[len(deltas)-1], _ = Hypothesis(thetas, trainingEx).Sub(trainingEx.ExpectedOutput)

	for i := len(deltas) - 2; i >= 0; i-- {
		workingTheta := thetas[i+1]

		levelPrediction := Hypothesis(thetas[:i+1], trainingEx).InsertRows(biasValueMatrix, 0)
		tmp, _ := matrix.Ones(levelPrediction.Rows(), 1).Sub(levelPrediction)
		levelGradient := levelPrediction.Dot(tmp)

		deltas[i] = workingTheta.Transpose().Mul(deltas[i+1]).Dot(levelGradient)
	}

	return deltas
}

// Helper functions

func sigmoidMatrix(index int, value float64) float64 {
	return sigmoid(value)
}

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Pow(math.E, -z))
}
