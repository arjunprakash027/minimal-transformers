package main

import (
	"fmt"
	"math/rand"
	"time"
)

type expr struct {
	format string
	eval func(a, b, c int) int
}

var expressions = []expr{
	{"%d + %d * %d = %d", func(a, b, c int) int { return a + b*c }},
	{"%d * %d + %d = %d", func(a, b, c int) int { return a*b + c }},
	{"%d - %d * %d = %d", func(a, b, c int) int { return a - b*c }},
	{"%d * %d * %d = %d", func(a, b, c int) int { return a * b * c }},
	{"%d + %d - %d = %d", func(a, b, c int) int { return a + b - c }},
	{"%d - %d - %d = %d", func(a, b, c int) int { return a - b - c }},
	{"%d + %d + %d = %d", func(a, b, c int) int { return a + b + c }},
	{"%d * (%d + %d) = %d", func(a, b, c int) int { return a * (b + c) }},
	{"(%d + %d) * %d = %d", func(a, b, c int) int { return (a + b) * c }},
}

func generateSample(r *rand.Rand) string {
	a := r.Intn(100)
	b := r.Intn(100)
	c := r.Intn(100)
	
	e := expressions[r.Intn(len(expressions))]
	result := e.eval(a,b,c)

	return fmt.Sprintf(e.format, a, b, c, result)
}

func main() {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	sample := generateSample(r)
	fmt.Println(sample)
}


