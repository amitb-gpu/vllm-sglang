package main

import "math/bits"

// pythonRandom implements the subset of CPython's random.Random needed by shuffle.
// CPython seeds MT19937 with a little-endian array of 32-bit words and uses
// rejection sampling on the high bits of each random word.
type pythonRandom struct {
	state [624]uint32
	index int
}

func newPythonRandom(seed uint32) *pythonRandom {
	rng := &pythonRandom{index: 624}
	state := &rng.state
	state[0] = 19650218
	for i := 1; i < 624; i++ {
		previous := state[i-1]
		state[i] = 1812433253*(previous^(previous>>30)) + uint32(i)
	}
	i := 1
	for range 624 {
		previous := state[i-1]
		state[i] = (state[i] ^ ((previous ^ (previous >> 30)) * 1664525)) + seed
		i++
		if i == 624 {
			state[0], i = state[623], 1
		}
	}
	for range 623 {
		previous := state[i-1]
		state[i] = (state[i] ^ ((previous ^ (previous >> 30)) * 1566083941)) - uint32(i)
		i++
		if i == 624 {
			state[0], i = state[623], 1
		}
	}
	state[0] = 0x80000000
	return rng
}

func (rng *pythonRandom) next() uint32 {
	if rng.index == 624 {
		for i := range rng.state {
			word := (rng.state[i] & 0x80000000) | (rng.state[(i+1)%624] & 0x7fffffff)
			rng.state[i] = rng.state[(i+397)%624] ^ (word >> 1)
			if word&1 != 0 {
				rng.state[i] ^= 0x9908b0df
			}
		}
		rng.index = 0
	}
	word := rng.state[rng.index]
	rng.index++
	word ^= word >> 11
	word ^= (word << 7) & 0x9d2c5680
	word ^= (word << 15) & 0xefc60000
	return word ^ (word >> 18)
}

func (rng *pythonRandom) below(n int) int {
	shift := 32 - bits.Len(uint(n))
	for {
		candidate := int(rng.next() >> shift)
		if candidate < n {
			return candidate
		}
	}
}

func workload(tickets []ticket, rounds int) []ticket {
	rng := newPythonRandom(seed)
	all := make([]ticket, 0, len(tickets)*rounds)
	for range rounds {
		shuffled := append([]ticket(nil), tickets...)
		for i := len(shuffled) - 1; i > 0; i-- {
			j := rng.below(i + 1)
			shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
		}
		all = append(all, shuffled...)
	}
	return all
}
