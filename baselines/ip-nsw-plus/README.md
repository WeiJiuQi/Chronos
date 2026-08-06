# ip-NSW+ baseline

This directory vendors the original GraphMIPS implementation of **ip-NSW+**
released by the authors of the AAAI 2020 paper *Understanding and Improving
Proximity Graph based Maximum Inner Product Search*.

- Upstream: <https://github.com/Jerry-liujie/ip-nsw/tree/GraphMIPS>
- Paper: <https://arxiv.org/abs/1909.13459>

ip-NSW+ augments an inner-product graph with an angular proximity graph used
to improve navigation. It indexes the mode-matched MIPS representation from
`tdvs_workload_builder/`: the weighted `dim`-dimensional base for
multiplicative TDVS, or the paired `dim+1` augmented base for additive TDVS.

## Chronos integration

Build the unified runner from the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 64 --target tdvs_ip_nsw_plus
```

The runner supports separate build, footprint, and search modes. Run

```bash
./build/baselines/tdvs_ip_nsw_plus --help
```

for its complete CLI. Invoke `--mode build`, `--mode search`, and
`--mode footprint` directly; [`../README.md`](../README.md) gives a complete
command sequence.

The transformed base remains unnormalized because its norm or final coordinate
carries the temporal signal. Multiplicative search uses the canonical semantic
query; additive search uses the exported augmented query. Query normalization
is performed outside the timed search interval and preserves MIPS ranking.

The default graph parameters are `M=32`, `efConstruction=1024`, angular
`M=10`, angular `efConstruction=100`, and angular `efSearch=1`. The main
`efSearch` list remains an explicit query-time input.

## Parallel construction

The vendored source includes a local thread-safety backport for the random
number generators and construction state used by parallel insertion. The
implementation is data-race-free under the covered construction tests, but
insertion interleaving can still change graph topology. Construction thread
count and source fingerprint are therefore recorded with every evaluation.
