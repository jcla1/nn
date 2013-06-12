package main

import (
	"fmt"
	"github.com/jcla1/matrix"
	"math"
)

func main() {
	//input := matrix.FromMatlab("[1;1]")
	thetas := []*matrix.Matrix{matrix.FromMatlab("[-30 20 20; 10 -20 -20]"), matrix.FromMatlab("[-10 20 20]")}

	//fmt.Println(hypothesis(thetas, input))

	//data := [][]*matrix.Matrix{[]*matrix.Matrix{matrix.FromMatlab("[1; 1]"), matrix.FromMatlab("[1]")}, []*matrix.Matrix{matrix.FromMatlab("[0; 1]"), matrix.FromMatlab("[0]")}, []*matrix.Matrix{matrix.FromMatlab("[1; 0]"), matrix.FromMatlab("[0]")}, []*matrix.Matrix{matrix.FromMatlab("[0; 0]"), matrix.FromMatlab("[0]")}}

	//fmt.Println(costFunction(data, thetas, 0.04))

	fmt.Println(deltaTerms(thetas, matrix.FromMatlab("[1; 1]"), matrix.FromMatlab("[1]")))

}
