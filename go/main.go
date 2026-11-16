package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
)

const seed = 5090
const warmup = "The April invoice lists our subscription twice. Please correct it."

type ticket struct {
	ID    string `json:"id"`
	Queue string `json:"queue"`
	Text  string `json:"text"`
}

type outcome struct {
	ID       string `json:"id"`
	Expected string `json:"expected"`
	*Route
	Error string `json:"error,omitempty"`
}

type run struct {
	Concurrency int       `json:"concurrency"`
	Requests    int       `json:"requests"`
	Successful  int       `json:"successful"`
	Errors      int       `json:"errors"`
	Correct     int       `json:"correct"`
	Accuracy    float64   `json:"accuracy_pct"`
	WallSeconds float64   `json:"wall_seconds"`
	RequestRate float64   `json:"successful_requests_per_second"`
	TokenRate   float64   `json:"output_tokens_per_second"`
	TTFT        *float64  `json:"p95_ttft_ms"`
	Latency     *float64  `json:"p95_latency_ms"`
	Outcomes    []outcome `json:"outcomes"`
}

type report struct {
	Engine        string `json:"engine"`
	URL           string `json:"url"`
	ModelAlias    string `json:"model_alias"`
	StartedAt     string `json:"started_at_utc"`
	DatasetSHA256 string `json:"dataset_sha256"`
	ShuffleSeed   int    `json:"shuffle_seed"`
	Rounds        int    `json:"rounds_per_level"`
	MaxTokens     int    `json:"max_output_tokens"`
	Runs          []run  `json:"runs"`
}

func readTickets(path string) ([]ticket, string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, "", err
	}
	var tickets []ticket
	ids := map[string]bool{}
	for _, line := range bytes.Split(bytes.TrimSuffix(data, []byte("\n")), []byte("\n")) {
		var item ticket
		if err := json.Unmarshal(line, &item); err != nil {
			return nil, "", err
		}
		if item.ID == "" || ids[item.ID] || strings.TrimSpace(item.Text) == "" || !slices.Contains(queues, item.Queue) {
			return nil, "", fmt.Errorf("invalid or duplicate ticket %q", item.ID)
		}
		ids[item.ID] = true
		tickets = append(tickets, item)
	}
	return tickets, fmt.Sprintf("%x", sha256.Sum256(data)), nil
}

func parallel(n, jobs int, fn func(int)) {
	work := make(chan int, jobs)
	for i := range jobs {
		work <- i
	}
	close(work)
	var workers sync.WaitGroup
	for range min(n, jobs) {
		workers.Go(func() {
			for i := range work {
				fn(i)
			}
		})
	}
	workers.Wait()
}

func rounded(value, scale float64) float64 { return math.RoundToEven(value*scale) / scale }

func p95(values []float64) *float64 {
	if len(values) == 0 {
		return nil
	}
	slices.Sort(values)
	result := rounded(values[int(math.Ceil(0.95*float64(len(values))))-1], 100)
	return &result
}

func measure(client *http.Client, url string, concurrency int, items []ticket) run {
	results := make([]outcome, len(items))
	start := time.Now()
	parallel(concurrency, len(items), func(i int) {
		item := items[i]
		results[i] = outcome{ID: item.ID, Expected: item.Queue}
		routing, err := triage(client, url, item.Text)
		if err != nil {
			results[i].Error = err.Error()
			return
		}
		results[i].Route = routing
	})
	seconds := time.Since(start).Seconds()
	summary := run{Concurrency: concurrency, Requests: len(items), Outcomes: results,
		WallSeconds: rounded(seconds, 1000)}
	var first, latency []float64
	tokens := 0
	for _, result := range results {
		if result.Route == nil {
			summary.Errors++
			continue
		}
		summary.Successful++
		if result.Expected == result.Queue {
			summary.Correct++
		}
		tokens += result.Tokens
		first, latency = append(first, result.TTFT), append(latency, result.Latency)
	}
	summary.Accuracy = rounded(100*float64(summary.Correct)/float64(summary.Requests), 100)
	summary.RequestRate = rounded(float64(summary.Successful)/seconds, 100)
	summary.TokenRate = rounded(float64(tokens)/seconds, 100)
	summary.TTFT, summary.Latency = p95(first), p95(latency)
	return summary
}

func execute(args []string) error {
	if len(args) == 0 || !slices.Contains([]string{"vllm", "sglang"}, args[0]) {
		return errors.New("usage: ticket-bench {vllm|sglang} --output PATH [--tickets PATH] [--concurrency 1,4,8]")
	}
	flags := flag.NewFlagSet(args[0], flag.ContinueOnError)
	output := flags.String("output", "", "benchmark JSON path")
	dataset := flags.String("tickets", "tickets.jsonl", "path to the baseline tickets")
	url := flags.String("url", defaultURL, "base URL ending in /v1")
	concurrency := flags.String("concurrency", "1,4,8", "comma-separated concurrency levels")
	rounds := flags.Int("rounds", 2, "workload repetitions per level")
	if err := flags.Parse(args[1:]); err != nil {
		return err
	}
	if *output == "" || *rounds <= 0 || flags.NArg() != 0 {
		return errors.New("require --output and positive --rounds")
	}
	var levels []int
	for _, field := range strings.Split(*concurrency, ",") {
		n, err := strconv.Atoi(field)
		if err != nil || n <= 0 {
			return fmt.Errorf("invalid concurrency %q", field)
		}
		levels = append(levels, n)
	}
	if err := os.MkdirAll(filepath.Dir(*output), 0755); err != nil {
		return err
	}
	items, digest, err := readTickets(*dataset)
	if err != nil {
		return err
	}
	client := &http.Client{Timeout: 60 * time.Second}
	ordered := workload(items, *rounds)
	result := report{Engine: args[0], URL: *url, ModelAlias: model,
		StartedAt: time.Now().UTC().Format(time.RFC3339Nano), DatasetSHA256: digest,
		ShuffleSeed: seed, Rounds: *rounds, MaxTokens: maxTokens}
	failed := false
	for _, level := range levels {
		warmupErrors := make([]error, level)
		parallel(level, level, func(i int) { _, warmupErrors[i] = triage(client, *url, warmup) })
		if err := errors.Join(warmupErrors...); err != nil {
			return fmt.Errorf("warmup failed: %w", err)
		}
		summary := measure(client, *url, level, ordered)
		result.Runs = append(result.Runs, summary)
		failed = failed || summary.Errors > 0
		fmt.Printf("concurrency=%d requests=%d errors=%d requests/s=%.2f\n", level, summary.Requests, summary.Errors, summary.RequestRate)
	}
	encoded, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(*output, append(encoded, '\n'), 0644); err != nil {
		return err
	}
	if failed {
		return errors.New("one or more measured requests failed")
	}
	return nil
}

func main() {
	if err := execute(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
