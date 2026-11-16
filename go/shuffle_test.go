package main

import (
	"encoding/json"
	"os"
	"testing"
)

func TestShuffleMatchesPublishedRun(t *testing.T) {
	tickets, digest, err := readTickets("../tickets.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("../results/vllm.json")
	if err != nil {
		t.Fatal(err)
	}
	var reference report
	if err := json.Unmarshal(raw, &reference); err != nil {
		t.Fatal(err)
	}
	if digest != reference.DatasetSHA256 || seed != reference.ShuffleSeed {
		t.Fatal("dataset hash or shuffle seed differs")
	}
	order := workload(tickets, reference.Rounds)
	expected := reference.Runs[0].Outcomes
	if len(order) != len(expected) {
		t.Fatalf("got %d tickets, want %d", len(order), len(expected))
	}
	for i, item := range order {
		if item.ID != expected[i].ID || item.Queue != expected[i].Expected {
			t.Fatalf("ticket %d = %s/%s, want %s/%s", i, item.ID, item.Queue, expected[i].ID, expected[i].Expected)
		}
	}
}
