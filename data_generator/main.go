package main

import (
	"fmt"
	"math/rand"
	"time"
	"encoding/csv"
	"os"
	"sync"
)

type expr struct {
	format string
	eval func(a, b, c int) int
}

type row struct {
	q string
	a string
}

var expressions = []expr{
	{"%d + %d * %d", func(a, b, c int) int { return a + b*c }},
	{"%d * %d + %d", func(a, b, c int) int { return a*b + c }},
	{"%d - %d * %d", func(a, b, c int) int { return a - b*c }},
	{"%d * %d * %d", func(a, b, c int) int { return a * b * c }},
	{"%d + %d - %d", func(a, b, c int) int { return a + b - c }},
	{"%d - %d - %d", func(a, b, c int) int { return a - b - c }},
	{"%d + %d + %d", func(a, b, c int) int { return a + b + c }},
	{"%d * (%d + %d)", func(a, b, c int) int { return a * (b + c) }},
	{"(%d + %d) * %d", func(a, b, c int) int { return (a + b) * c }},
}

func generateSample(r *rand.Rand) (string, string) {
	a := r.Intn(100)
	b := r.Intn(100)
	c := r.Intn(100)
	
	e := expressions[r.Intn(len(expressions))]
	result := e.eval(a,b,c)
	
	question := fmt.Sprintf(e.format, a, b, c)
	answer := fmt.Sprintf("%d", result)

	return question, answer
}

func worker(
	id int, 
	jobs <-chan struct{},
	out chan<- row,
	wg *sync.WaitGroup,
){
	defer wg.Done()

	r := rand.New(rand.NewSource(time.Now().UnixNano() + int64(id)))

	for range jobs {
		q, a := generateSample(r)
		out <- row{q, a}
	}

}


func main() {
	
	const (
		numSamples = 1000000
		numWorkers = 8
	)

	file, err := os.Create("bodmas.csv")

	if err != nil {
		panic(err)
	}

	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	writer.Write([]string{"question", "answer"})
	
	jobs := make(chan struct{}, 1000)
	out := make(chan row, 1000)

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for i := 0; i < numWorkers; i++ {
		go worker(i, jobs, out, &wg)
	}

	go func() {
		for i := 0; i < numSamples; i++ {
			jobs <- struct{}{}
		}
		close(jobs)
	}()

	go func(){
		wg.Wait()
		close(out)
	}()

	for r := range out {
		writer.Write([]string{r.q, r.a})
	}
	
}


