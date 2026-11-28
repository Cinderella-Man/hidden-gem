# NCCL with memory sharing: an NCCL fork for pairs CUDA cannot connect

> **EXPERIMENTAL.** On a Tesla V100 (in a PCIe 3.0 x4 chipset slot) + RTX 3090 (PCIe 4.0 x16) this fork moves large
> all-reduces **44 % faster** than NCCL does through host RAM and makes plain llama.cpp **29 % faster** on prompts. Most
> of llama.cpp's gain on this pair comes from llama.cpp itself, on stock NCCL (the
> [P2P guide's llama.cpp section](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus#llamacpp-build-it-with-nccl)):
> on top of those changes the fork adds **5 %** (974 -> 1,017-1,028 tokens/s), all from the large all-reduces of prompt
> processing: small ones keep NCCL's own path. Since then also run on three RTX 3090s for four hours and in rings over
> three, four and five GPU generations ([three-generations.md](three-generations.md)), every answer identical to stock
> NCCL. Since 2026-11-27 the fork checks every pair of cards before using it (a pair that fails goes through host memory)
> and uses the safe `readback` ordering by default; the faster `release` of the tables below is an opt-in for machines
> where every card's link is fast ([safe by default](#safe-by-default-every-pair-checked-readback-everywhere)).

NVIDIA's NCCL moves data between GPUs for CUDA programs. Between two cards that CUDA refuses to pair, like a Tesla V100
and an RTX 3090 (two GPU generations: no peer access), NCCL has no direct route and goes through host RAM. This page
builds **a fork of NCCL 2.32.6 with one more transport**: it moves NCCL's data card to card through DRMShare's shared
memory, with the cards' copy engines (through Vulkan), while CUDA still sees no peer access. Programs need no change:
the fork is a drop-in `libnccl.so.2`.

## Results

Tesla V100 (PCIe 3.0 x4, a slot wired through the board's chipset) + RTX 3090 (PCIe 4.0 x16), NVIDIA's closed driver
580.178.04 with DRMShare's `nvidia-drm`, NCCL 2.32.6, 2026-11-26. Every NCCL result checked; every llama.cpp
comparison with identical perplexity, chunk for chunk.

NCCL's `all_reduce_perf`, two cards:

| | 8 B | 2 MiB | 8 MiB | 128 MiB |
|---|---|---|---|---|
| NCCL (shared memory in host RAM) | 13.6 µs | 1.75 GB/s | 1.80 GB/s | 1.82 GB/s |
| this fork, `readback` with the V100 in `NCCL_DRMSHARE_NOREAD` (it fetches its data itself) | 13.0 µs | 1.52 GB/s | 1.94 GB/s | 1.94 GB/s |
| **this fork, `release`** (an opt-in since 2026-11-27: fast links only) | 13.0 µs | **2.18 GB/s** | **2.60 GB/s** | **2.63 GB/s** |

The ceiling is the V100's PCIe 3.0 x4 slot with both directions busy, about 2.8 GB/s: these numbers measure the slot as
much as the method. Two cards in faster slots move several times more (13.1 GB/s between two RTX 3090s on PCIe 4.0 x8
lanes, [dmabuf-615.md D5](dmabuf-615.md#d5-three-rtx-3090s)). Small all-reduces go through NCCL's own
shared memory either way (the fork's two lanes, below).

llama.cpp, Qwen3.8-27B Q8_0, `-sm tensor`, prompt 2048 tokens per second, 2 rounds each:

| llama.cpp | stock NCCL | **fork (`release`)** | fork, `readback` + `NOREAD` | generation, all |
|---|---|---|---|---|
| plain (upstream as of 2026-11-24) | 720 | **932-933 (+29 %)** | 836 (+16 %) | 33.1-33.3 |
| with its tensor-parallel speed-ups, `-ts 3/1 -ub 2048` (best prompts) | 974-975 | **1,017-1,028 (+5 %)** | 931-937 (-3 %, older build) | 33.7-33.8 |
| the same at `-ts 3/2` (best generation), `readback` | 921-927 | | 909-910 (-1.6 %) | 37.4 |
| the same at `-ts 3/2`, 8k / 32k-token prompts, `readback` | 821 / 552 | | 818 / 550 | 36.0 at 8k, 34.1-34.6 at 32k |

With llama.cpp's speed-ups, moving data overlaps with computing, and what counts is how little the transfer disturbs
the GPUs as well as its speed. With the V100 fetching its own data (`NOREAD`: its copy engine then works both
directions) `readback` cost more there than it saved; `release` (every card writes its own data, no read across) came
out ahead. Both links in this machine are fast, where `release` is safe. `readback` without `NOREAD`, the default now,
was not measured in this layout; on the same two cards with CPU lanes it was 0.5 % behind `release`
([safe by default](#safe-by-default-every-pair-checked-readback-everywhere)).

## What you need

- **A pair CUDA cannot connect.** If CUDA already offers peer access between all your cards (the P2P guide's patched
  driver, or its patched copy of `libcuda` for two generations), NCCL's own P2P transport comes before this fork's and
  the fork changes nothing: use stock NCCL. An outside report with P2P on, an RTX 3060 + RTX 2070
  ([#1](https://github.com/Cinderella-Man/drmshare/issues/1)), got `via P2P/direct pointer` from the fork, as stock
  NCCL gives. Without `NCCL_DRMSHARE_TRANSPORT=1` (step 3) the fork is stock NCCL anyway.
- **DRMShare working on your pair**: its `nvidia-drm` loaded with `peer_vidmem_dmabuf=1`, and its copy test passing
  ([volta-dmabuf.md](volta-dmabuf.md) for a V100, Pascal or Maxwell card, [dmabuf-615.md](dmabuf-615.md) for Turing and newer).
  Check: `cat /sys/module/nvidia_drm/parameters/peer_vidmem_dmabuf` prints `Y`.
- **Vulkan on both cards**: the Vulkan loader and NVIDIA's Vulkan driver (`vulkaninfo --summary` lists both GPUs), and
  the headers to build: `sudo apt install libvulkan-dev`.
- **A CUDA toolkit that builds for both cards.** A Tesla V100 (sm_70), a Pascal card or a Maxwell card (a GTX 750 Ti is
  sm_50) needs CUDA 12.x: CUDA 13 dropped Maxwell, Pascal and Volta. Ubuntu's
  `nvidia-cuda-toolkit` (12.4 on 26.04) works.
- **One program that drives both GPUs from one process** (llama.cpp does). Programs with one process per GPU are not
  supported yet.

## 1. Build the NCCL fork

The fork is pinned here as `tools/nccl-drmshare` ([Cinderella-Man/nccl-drmshare](https://github.com/Cinderella-Man/nccl-drmshare),
branch `2.32.6-drmshare`: three commits on NVIDIA's `v2.32.6-1` tag: the transport, its pair check and `readback` default, the channel mix).

```bash
git submodule update --init --depth 1 tools/nccl-drmshare
cd tools/nccl-drmshare
make -j8 src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_86,code=sm_86"
```

Put your cards' compute capabilities in `NVCC_GENCODE` (5.0 for a GTX 750 Ti, 7.0 for a V100, 7.5 Turing, 8.6 RTX 30,
8.9 RTX 40). With
Ubuntu's toolkit instead of NVIDIA's (`/usr/local/cuda`), add `CUDA_HOME=/usr CUDA_LIB=/usr/lib/x86_64-linux-gnu
CUDA_INC=/usr/include`, and `CUDARTLIB=cudart` if `/usr/lib/x86_64-linux-gnu/libcudart_static.a` is missing. To build
only what llama.cpp uses (much faster), add `ONLY_FUNCS="AllReduce Sum f32 * *|AllReduce Sum bf16 * *|SendRecv"`.
The library lands in `build/lib/libnccl.so.2.32.6`.

## 2. Build llama.cpp with NCCL

Any llama.cpp with the CUDA backend's NCCL support builds against the fork as against NCCL itself (same ABI), for
example the copy pinned here as `tools/llama.cpp` ([llama.cpp-drmshare](https://github.com/Cinderella-Man/llama.cpp-drmshare),
which also carries the tensor-parallel changes of [step 3's faster run](#faster-llamacpp-drmshares-tensor-parallel-changes)):

```bash
cmake -S tools/llama.cpp -B tools/llama.cpp/build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_NCCL=ON \
  -DNCCL_ROOT=$PWD/tools/nccl-drmshare/build -DCMAKE_CUDA_ARCHITECTURES="70;86"
cmake --build tools/llama.cpp/build-cuda -j8 --target llama-bench llama-perplexity llama-server
```

CMake's output must say `Found NCCL`: without it llama.cpp builds without NCCL, silently.

## 3. Run

```bash
export LD_LIBRARY_PATH=$PWD/tools/nccl-drmshare/build/lib
export NCCL_DRMSHARE_TRANSPORT=1
export NCCL_DRMSHARE_SPLIT_BYTES=4194304    # 2097152 with NCCL_DRMSHARE_ORDER=release
tools/llama.cpp/build-cuda/bin/llama-bench -m model.gguf -ngl 99 -sm tensor -p 2048 -n 128
```

| Setting | What it does |
|---|---|
| `NCCL_DRMSHARE_TRANSPORT=1` | turns the transport on. With none of the `NCCL_DRMSHARE_` variables set, the fork behaves as NCCL 2.32.6 |
| `NCCL_DRMSHARE_SPLIT_BYTES=<bytes>` | **two lanes**: all-reduces below go through NCCL's shared memory (about 13 µs for a small one), larger ones card to card. llama.cpp's generation makes about 128 small all-reduces per token; without the cut each pays the copy engine's 28-45 µs. Set it only together with `NCCL_DRMSHARE_TRANSPORT=1`: alone, it sends large transfers through NCCL's socket network |
| `NCCL_DRMSHARE_ORDER` | `readback` (the default since 2026-11-27): after each copy, 4 bytes are read back from the destination (a PCIe read cannot pass the writes before it); **0 wrong on every machine and link tested, slow ones included**; 0.5 % behind `release` in llama.cpp on a V100 + RTX 3090. `release` (the default until then): every card writes its own data into the other, each copy followed by Vulkan's release of that memory to the other device; reads nothing across. **`release` lets wrong data through whenever a link that carries peer writes is slow, in this program or another one on the same machine** (a USB x1 riser, a port running at 2.5 GT/s: [safe by default](#safe-by-default-every-pair-checked-readback-everywhere)), and it did on a GTX 750 Ti behind a chipset ([Maxwell](#maxwell-a-gtx-750-ti-with-an-rtx-3090)). Use it only where every card's link runs at 8 GT/s and x4 or wider, and after the stale-step test ([5](#5-check-that-the-results-are-right)) passed with it on your cards; the fork warns when it sees a slow link on a pair with `release` set. `none`: no ordering, for reproducing the first version's bug only |
| `NCCL_DRMSHARE_NOREAD=<bus id>` | `readback` only: a card whose memory must not be read by the other card fetches its data itself (slower: its copy engine then works both directions). For a **Tesla V100 in a chipset slot**, which can freeze when another card reads its memory in large bursts ([troubleshooting](troubleshooting.md#a-tesla-v100-in-a-chipset-slot-freezes-xid-62-while-another-card-reads-its-memory)): `readback` reads 4 bytes per step there, far from the 1 MiB bursts that stopped it (4 and 64 KiB bursts never did), but that layout was not tested with the read-backs: set it to be sure. `release` never reads across. The bus id from `nvidia-smi --query-gpu=name,pci.bus_id --format=csv`, any unique part of it (`04:00`) |
| `NCCL_DRMSHARE_CHANNEL_MIX=<d>:<s>` | `d` of every `d+s` channels of the large lane card to card, the other `s` through host memory. Where every card has its own CPU link, `1:3` with `NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4` made the fork faster than stock NCCL ([both routes at once](#faster-on-cpu-lanes-both-routes-at-once)); unset, every channel goes card to card |
| `NCCL_DRMSHARE_CHECK=0` | skips the pair check's writes (each card writing 4 MiB into the other before the first connection, [safe by default](#safe-by-default-every-pair-checked-readback-everywhere)); a pair that cannot reach the other then gives wrong results instead of going through host memory |
| `NCCL_DRMSHARE_TEST_REFUSE=<a>,<b>` | for testing the fallback: refuses the pair of CUDA devices `a` and `b` as if its check had failed |

### Faster: llama.cpp-drmshare's tensor-parallel changes

The copy pinned as `tools/llama.cpp` carries four changes to llama.cpp's tensor parallelism for cards linked through
host memory or a slow slot; they change nothing unless their settings are on, and they help with stock NCCL too (the
P2P guide's [llama.cpp section](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus#more-speed-tensor-parallel-changes-in-a-llamacpp-copy-experimental) explains each). With the fork's
settings above, add:

```bash
export GGML_CUDA_ALLREDUCE=split   # small all-reduces by llama.cpp's own path, large ones by NCCL
export GGML_META_ASYNC_AR=1        # a layer's all-reduces overlap the next part of its work
tools/llama.cpp/build-cuda/bin/llama-bench -m model.gguf -ngl 99 -sm tensor -ts 3/1 -ub 2048 -p 2048 -n 128
```

This is the run behind the 1,017-1,028 tokens/s in [Results](#results) (974-975 with stock NCCL; Qwen3.8-27B Q8_0 on
the V100 + RTX 3090, 33.7 generating).

- `-ts 3/1` gives three quarters of each layer to CUDA device 0. CUDA numbers the fastest card first, here the RTX
  3090; llama.cpp lists the devices in that order when it starts. Split by your cards' speed: on this pair `-ts 3/1`
  is the best for prompts and `-ts 3/2` for generation (37.4 tokens/s, Results).
- On cards with different amounts of memory, split by memory first: without `-ts`, `-sm tensor` gives every card an
  equal share of each layer, so the smaller card runs out first, and llama.cpp's automatic fit (`--fit`) does not
  handle `-sm tensor`. llama.cpp-drmshare warns about it when it loads and prints a split by memory.
- The overlap needs batches of 1,024 tokens or more (`-ub 1024` or `-ub 2048`) and cuts each layer into two chunks of
  tokens, so far only for the Qwen3.5 / 3.8 family (`qwen35` in llama.cpp). Other models still get the two lanes.

For a server, the same settings with the split for chat (at `-ts 3/1` the RTX 3090 has no room left for a 32k context
with this model):

```bash
tools/llama.cpp/build-cuda/bin/llama-server -m model.gguf -ngl 99 -sm tensor -ts 3/2 -ub 2048 -b 2048 -c 32768
```

llama-server's speed was not measured: the numbers on this page are llama-bench's.

## 4. Check that it is used

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT tools/llama.cpp/build-cuda/bin/llama-bench -m model.gguf -ngl 99 -sm tensor -p 2048 -n 16 -r 1 2>&1 | grep -E 'DRMShare|via'
```

You should see one line per card (`DRMShare: CUDA device 0 = Vulkan NVIDIA GeForce RTX 3090, copy queues: ...`), one
per pair of cards (`DRMShare: CUDA devices 0 (0000:2C:00.0) and 1 (0000:2D:00.0): writes and read-backs across checked
both ways; slowest links 8.0 GT/s x4 ..., ordering readback`) and connection lines `via DRM`, made at the first large
all-reduce; the small lane's lines say `via SHM/direct`. A pair that failed the check has a warning saying why, and
its large-lane connections say `via SHM/direct` too. No `via DRM` at all: check `NCCL_DRMSHARE_TRANSPORT` and that
DRMShare's module is loaded (the warnings say which).

## 5. Check that the results are right

The transport has to get PCIe's ordering right: a copy engine reports "done" when its writes have left, and on a link
with relaxed ordering the "data ready" mark could be seen before the data. The first version had exactly that bug
(one step read early). Check on your own cards:

- **NCCL's own test**, every result compared: build [nccl-tests](https://github.com/NVIDIA/nccl-tests) against the fork
  (`make MPI=0 NCCL_HOME=$PWD/tools/nccl-drmshare/build`), then `./build/all_reduce_perf -g 2 -b 8K -e 256M -f 2 -n 50 -c 1`:
  the `#wrong` columns must be 0.
- **Back to back**: nccl-tests' check starts each checked all-reduce from idle and did not catch the first version's
  bug (0 wrong in 7000 checks). Thousands of all-reduces back to back did: the first version gave 79 424 wrong
  elements at 64-512 KiB, the `readback` and `release` orderings 0 in 42 000 all-reduces each (two runs, 64 KiB-4 MiB,
  3000 per size). [`tools/stale_test.cu`](../tools/stale_test.cu): `nvcc -O2 -arch=native stale_test.cu -o stale_test
  -I<nccl>/build/include -L<nccl>/build/lib -lnccl`, then `./stale_test 65536 4194304 3000` with the fork's settings
  (add `NCCL_SHM_DISABLE=1 NCCL_NET_DISABLE_INTRA=1 NCCL_PROTO=Simple` and leave out the split to send every size through the transport).
- **llama.cpp's perplexity** with and without the `NCCL_DRMSHARE_` variables: with two cards the sums come out the same
  in either order, so every chunk should be identical.

## Maxwell: a GTX 750 Ti with an RTX 3090

The fork on a GTX 750 Ti (GM107, Maxwell, 2 GB, 256 MB window) in a PCIe 3.0 x4 chipset slot and an RTX 3090 (PCIe 4.0
x16), NVIDIA's closed 580.178.04 with DRMShare's `nvidia-drm` ([volta-dmabuf.md](volta-dmabuf.md#gtx-750-ti--rtx-3090-maxwell--ampere-no-v100)),
2026-11-27. Built with Ubuntu's CUDA 12.4 for `compute_50` + `sm_86` (NCCL builds for sm_50; nccl-tests' all-to-all
test needs sm_60 and up, so build `all_reduce_perf` alone: `make -C src MPI=0 ... ../build/all_reduce_perf`).

- **The transport comes up on the Maxwell card** (`DRMShare: CUDA device 0 = Vulkan NVIDIA GeForce GTX 750 Ti`,
  connections `via DRM`). `all_reduce_perf`, every result checked, 0 wrong:

  | | 8 B | 2 MiB | 8 MiB | 128 MiB |
  |---|---|---|---|---|
  | NCCL (shared memory in host RAM) | 13.7 µs | 1.75 GB/s | 1.82 GB/s | 1.85 GB/s |
  | **this fork, two lanes** | 13.5 µs | **2.24 GB/s** | **2.62 GB/s** | **2.66 GB/s** |
  | this fork, every all-reduce card to card | 122.7 µs | 2.37 GB/s | 2.62 GB/s | 2.66 GB/s |

  The same as the V100 + RTX 3090 in that slot: the slot is the ceiling.
- **`release` (the default then) was not safe on this card.** The back-to-back test with every all-reduce card to card
  (`stale_test`, fresh process each run, 3000 per size): with `release`, **wrong values twice in about 50 runs**, 64
  and 1,152 elements, each time at 64 KiB, the first all-reduces of a fresh connection, in the 750 Ti's results; with
  **`readback`, 0 wrong in 48 runs** (over 480,000 all-reduces, 59 billion elements checked). The cause is not known
  (the 750 Ti's writes through its window seen out of order?); the V100 + RTX 3090 never showed it (above). This and
  the slow-link failures made `readback` the default ([safe by default](#safe-by-default-every-pair-checked-readback-everywhere)).
- **llama.cpp's CUDA backend runs across the pair** (llama.cpp-drmshare, `-sm tensor`): Qwen3-0.6B perplexity identical
  through the fork and through stock NCCL, chunk for chunk; Qwen3.8-27B UD-Q3_K_XL split 1:23 with MTP and DFlash2
  drafting, every answer identical through the fork and through stock NCCL. Use `-ub 512`: at 2048 the 750 Ti's 2 GB
  runs out. It is slow, though: the 750 Ti's compute holds every step back.

  | Qwen3.8-27B UD-Q3_K_XL, tokens/s | RTX 3090 alone | 750 Ti + 3090, NCCL | 750 Ti + 3090, this fork |
  |---|---|---|---|
  | llama-bench, prompt 512 / generation | 1,533 / 47.2 | 260 / 16.7 | 280 / 16.7 |
  | llama-server, MTP 3 | 79.3 | 19.0 | 19.0 |
  | llama-server, DFlash2 draft | 98.3 | 22.5 | 22.5 |

  A coverage result: a Maxwell card works with the fork, it does not make a 3090 faster.

## Three RTX 3090s on driver 615: a four-hour stress

The fork's first run beyond two cards, and its first on driver 615: three RTX 3090s (one in a PCIe 4.0 x4 chipset slot,
two on PCIe 4.0 x8 CPU lanes, 250 W caps) on Ubuntu's stock `nvidia.ko` 615.71.09 with DRMShare's 615 `nvidia-drm`
([dmabuf-615.md](dmabuf-615.md)), no CUDA P2P, 2026-11-27. NCCL builds a ring 0 -> 1 -> 2 -> 0 `via DRM`, the chipset
card included. llama.cpp-drmshare with Qwen3.8-27B Q8_0 and Qwen3.8-Flash-Next IQ3_S, `llama-server` sessions of six
requests (one of them an 8,000-token prompt), temperature 0, every answer compared byte for byte; 4 h 12 min:

| Setting | Sessions | Generation, tokens/s | Answers vs stock NCCL |
|---|---|---|---|
| 27B, MTP 3 (its own nextn layer), two lanes | 11 | 85.1-85.5 | **132 of 132 identical** |
| 27B, plain, two lanes | 11 | 49.8-50.0 | **132 of 132 identical** |
| 27B, DFlash2 draft (7 tokens), tensor split, two lanes | 10 | 89.4-89.6 | identical (control) |
| Flash-Next, plain, two lanes | 12 | 70.1-70.2 | **144 of 144 identical** |
| Flash-Next, DFlash2 draft (3), tensor split, two lanes | 12 | 114.0-114.2 | **144 of 144 identical** |
| every all-reduce card to card (both models) | 23 | 43.8-46.5 | differ (see below) |

- **No failure** in 95 sessions (the 96th was the DFlash2 attempt before the draft fix below), no hang, nothing in the
  kernel log (Xid, PCIe errors, `nvidia-drm` errors, GSP timeouts). Every session repeated its setting's first answers
  byte for byte (1,140 answers).
- **The back-to-back test** (`stale_test` for N cards, every element of every all-reduce checked on the GPU, card to card
  64 KiB-4 MiB and the two lanes 1-16 MiB, f32 and bf16): about 1.66 million all-reduces, 6.37 trillion elements,
  **0 wrong**. Perplexity over 16 chunks identical to stock NCCL chunk for chunk, both models.
- **Speed**: 27B prompts (`llama-bench -sm tensor -ub 2048`) **1,814 tokens/s against 1,492 with stock NCCL (+22 %)**,
  96 % of real CUDA P2P on the same cards (1,880-1,891); generation the same as stock (51.3: the shared-memory lane).
  Long prompts in the server +16-18 % on the 27B, +6-7 % on Flash-Next (a mixture of experts: less all-reduce per token
  of compute).
- **Every all-reduce card to card** (`NCCL_SHM_DISABLE=1 NCCL_NET_DISABLE_INTRA=1 NCCL_PROTO=Simple`) costs 37-46 % of
  generation (the copy engine's latency per small transfer), so keep the two lanes on three cards too. Its answers
  differ from stock NCCL's default, yet are identical to stock NCCL forced to the same Simple protocol (6 of 6 on each
  model): the protocol sums in another order; the data arrives intact.
- **DFlash2 with `-sm tensor`** needs a draft with its own token embeddings and output head. A draft without them
  borrows the target's, which tensor split cuts by vocabulary across the cards, and llama.cpp aborts
  (`ggml-backend-meta.cpp:578`). [`scripts/dflash_add_head.py`](../scripts/dflash_add_head.py) copies the two tensors
  from the target into the draft (pure Python, byte for byte); then run the draft on one card (`-devd CUDA1`) with
  `LLAMA_DFLASH_DEV=0` (the target's hidden states reach the draft by host copies).

## The V100 + RTX 3090 with the slots swapped

The numbers above for this pair are with the V100 in the PCIe 3.0 x4 chipset slot and the RTX 3090 in the x16 CPU
slot. A reader asked for the other way round: the RTX 3090 in the chipset slot, the V100 in the CPU slot (PCIe 3.0
x16). Same scripts, builds and settings, Qwen3.8-27B Q8_0, tokens/s, prompt 2048 / generation:

![Tesla V100 + RTX 3090, llama.cpp routes in both slot layouts (prompt 2048 / generation, tokens/s). V100 x4 and RTX 3090 x16 against RTX 3090 x4 and V100 x16: 1 Vulkan through host RAM 347 / 27.9 against 474 / 30.2; 2 Vulkan + DRMShare 412 / 28.9 against 388 / 30.3; 3 CUDA, llama.cpp's own all-reduce 647 / 34.7 against 787 / 36.2; 4 CUDA + NCCL 720 / 33.2 against 878 / 35.2; 5 CUDA + this fork 932 / 33.1 against 926 / 35.2; 6 llama.cpp-drmshare's changes with NCCL, -ts 3/1, 974-984 / 33.8 against 992-1,037 / 31.6; 7 the same with this fork 1,017-1,028 / 33.7 against 1,033-1,040 / 31.6; 8 the changes at -ts 3/2 925 / 37.3 against 1,035-1,040 / 36.2](assets/llama-routes-v100-3090-swapped.png)

- **Every route through host RAM gains** (+22-37 % on prompts): the V100, the slower card, now moves its data over
  x16. Routes that copy card to card still cross the RTX 3090's x4 link, so they gain nothing, and **this fork's lead
  over stock NCCL shrinks from +29 % to +5 % or less**.
- **Best in this layout: llama.cpp-drmshare's changes with stock NCCL at `-ts 3/2`, about 1,038 / 36.2**, one setting
  for prompts and chat. Perplexity: the fork identical to stock NCCL chunk for chunk, in both layouts.
- So with a V100 and a GeForce card: put the V100 in the x16 slot if you can, and use stock NCCL; the fork pays off
  when the slower card has to sit behind the narrow link.

## Three generations: a V100, a GTX 1650 SUPER and an RTX 3090

One machine, NVIDIA's closed 580.178.04 with DRMShare's `nvidia-drm`, 2026-11-27: a Tesla V100 (Volta) in the CPU slot
(PCIe 3.0 x16), an RTX 3090 (Ampere) in the chipset's x16-size slot (PCIe 3.0 x4), and a GTX 1650 SUPER (Turing, TU116,
4 GB) on a USB-cable x1 riser in a chipset x1 slot, held at 2.5 GT/s
([troubleshooting](troubleshooting.md#a-card-on-a-pcie-x1-riser-falls-off-the-bus-xid-79-when-the-driver-starts-it)),
so about 0.25 GB/s each way. Built with Ubuntu's CUDA 12.4 for `sm_70;75;86`.

**The fork cannot run all three here.** NCCL's ring on three cards uses every pair, and the RTX 3090 and the GTX 1650
SUPER, both behind the chipset, cannot reach each other at all
([troubleshooting](troubleshooting.md#two-cards-behind-one-amd-chipset-the-copy-finishes-nothing-arrives)). With all
three cards, use stock NCCL (shared host memory), which never needs one card to reach another.

**On the V100 + GTX 1650 SUPER pair** (both reached through the CPU; `CUDA_VISIBLE_DEVICES` hiding the 3090):

- The transport comes up (`DRMShare: CUDA device 0 = Vulkan NVIDIA GeForce GTX 1650 SUPER`, connections `via DRM`).
- **`release` ordering let wrong values through on every run**: up to 6.5 % of the elements at 64-256 KiB, on both
  cards. **`readback`: 0 wrong** in every check below. A likely reason is the slow link, which widens the gap between
  the data landing and the "data ready" mark; the V100 + RTX 3090 at x4 never showed it.
- With `NCCL_DRMSHARE_ORDER=readback` and `NCCL_DRMSHARE_SPLIT_BYTES=4194304`: the back-to-back test card to card
  (64 KiB-4 MiB x 3,000) and through the two lanes (4-16 MiB x 300), f32 and bf16, **0 wrong**; `all_reduce_perf`
  with every result checked, 0 wrong (0.13 GB/s for stock NCCL, 0.18-0.19 GB/s for the fork: the riser's limit).
- llama.cpp-drmshare, Qwen3.8-27B UD-Q3_K_XL split 1:11 (the 1650 SUPER's 4 GB): perplexity **identical** to stock NCCL
  chunk for chunk; `llama-server` with MTP 3 and with a DFlash2 draft, **every answer identical** to stock NCCL (5 of 5
  each, with the same drafted and accepted counts).

  | Tokens/s, prompt 2048 / generation | V100 alone | V100 + 1650 SUPER, NCCL | V100 + 1650 SUPER, this fork |
  |---|---|---|---|
  | llama-bench | 715 / 33.9 | 80 / 21.5 | **105 / 21.5** |
  | llama-server, MTP 3 | | 19.9 | 19.9 |
  | llama-server, DFlash2 draft | | 23.5 | 23.4 |

  The fork is 32 % ahead of stock NCCL on prompts, but any split over a 0.25 GB/s riser is far slower than the V100
  alone: like the GTX 750 Ti, a coverage result.

**All three generations in one llama.cpp process** (stock NCCL through host memory for CUDA, host RAM for Vulkan; the
RTX 3090 at its stock 350 W), Qwen3.8-27B Q8_0, tokens/s, prompt 2048 / generation:

| Route | RTX 3090 + V100 | RTX 3090 + GTX 1650 SUPER + V100 |
|---|---|---|
| CUDA + NCCL, tensor split (`-ts 12/15`, `12/1/15`) | 844 / 33.1 | 75 / 19.1 |
| CUDA, layer split | 1,268 / 25.3 | 558 / 22.9 |
| Vulkan through host RAM, tensor split | 441 / 29.0 | 47 / 13.2 |
| Vulkan, layer split | | 468 / 9.0 |

It runs, and the answers hold: perplexity over 8 chunks 5.9998 on three cards against 5.9984 on two (CUDA; another
order of summing), 6.0008 on Vulkan; nothing in the kernel log. The third card only costs speed here, because every
all-reduce waits for its 0.25 GB/s riser; an outside report on an Intel board found three cards slower than two as well.

**The same three cards on CPU lanes: one ring across three generations** (with the other sets, GTX 1070 Ti or GTX 750
Ti + RTX 3090 + V100, and a ring over four generations, in [three-generations.md](three-generations.md)). The same day the board was rebuilt with a
bifurcation splitter in the CPU slot (x16 -> 4 x x4, Gen 3; the board's lane setting x4x4x4x4 and its speed Gen3): each
card on its own CPU port at PCIe 3.0 x4, nothing behind the chipset. Now every pair reaches every other
(memory sharing 3.30 GB/s each way, all six directions at once 196 GB checked, 0 wrong), and the fork builds one ring
over Turing, Ampere and Volta (`DRMShare: CUDA device 0 = Vulkan NVIDIA GeForce GTX 1650 SUPER`, `1 = ... RTX 3090`,
`2 = Tesla V100`, six connections `via DRM`):

- **Correct**: the back-to-back test card to card and through the two lanes, f32 and bf16, 0 wrong; `all_reduce_perf`
  over three cards, every result checked, 0 wrong; perplexity identical to stock NCCL chunk for chunk; `llama-server`
  with MTP 3 and with the DFlash2 draft, every answer identical (5 of 5 each). **`release` held here too** (0 wrong,
  3,000 all-reduces per size): on the slow riser it failed, at x4 it did not.
- **But not faster**: with every card on its own CPU link, NCCL's route through host memory is the better one.

  | Qwen3.8-27B Q8_0, tokens/s, prompt 2048 / generation | stock NCCL | this fork |
  |---|---|---|
  | `all_reduce_perf`, three cards, 8-128 MiB | 2.33-2.38 GB/s | 1.51-1.54 GB/s (`readback` or `release`) |
  | RTX 3090 + V100 | 838 / 32.2 | 857 / 32.1 |
  | all three | **460 / 30.8** | 405 / 30.8 |

  One pair alone copies at 3.30 GB/s, but in a ring every card sends and receives at once, and all the card-to-card
  traffic turns at the CPU (about 1.5 GB/s per card here), while each card's own x4 link to host memory carries the
  shared-memory route at 2.4. **The fork pays off where a card's path to host memory is the bottleneck** (a card in a
  chipset slot: +29 % on the V100 + RTX 3090), **not when every card has CPU lanes**: measure both on your machine.

## Safe by default: every pair checked, `readback` everywhere

Since 2026-11-27 the fork does two things on its own that earlier needed the user:

- **It checks every pair before using it.** Before the first connection between two cards, each card writes 4 MiB into
  the other's memory, reads the end of it back across, and the other card checks its own memory word for word. A pair
  that fails is named in a warning and **goes through host memory** instead: two cards behind one chipset (their writes
  to each other vanish without an error), or a card whose memory cannot be imported (DRMShare's module not loaded).
  This works on the two lanes' large lane (`NCCL_DRMSHARE_SPLIT_BYTES`); with every size card to card and
  `NCCL_SHM_DISABLE=1` there is no other route, and NCCL stops with an error instead of giving wrong results.
- **It uses `readback` for every pair.** `release` stays as an opt-in (`NCCL_DRMSHARE_ORDER=release`), for machines
  where every card's link is fast.

Tested on four cards in one machine, NVIDIA's closed 580.178.04: a GTX 1070 Ti (Pascal) behind the B550 chipset (PCIe
3.0 x4), and an RTX 2060 (Turing), an RTX 3090 (Ampere) and a Tesla V100 (Volta) on a bifurcation splitter's CPU ports
(PCIe 3.0 x4 each).

- **The check** ran on the four pairs NCCL's ring connects, writes and read-backs both ways, all passed; it adds about
  0.15 s per pair to the start.
- **The fallback**: one pair refused on purpose (`NCCL_DRMSHARE_TEST_REFUSE`, a test setting): that pair's two
  connections went `via SHM/direct`, the other six `via DRM`, 0 wrong. Every pair refused for real (DRMShare's option
  switched off, so no card can import another's memory): the warning names the cause, everything goes through host
  memory at stock NCCL's speed, 0 wrong. Before this change the first was a hard error at the first large all-reduce,
  and a pair behind one chipset gave wrong results without any error.
- **Why `readback` for every pair.** `release` goes wrong whenever any link that carries peer writes is slow, also on
  pairs whose own links are fast. With the RTX 2060's port held at 2.5 GT/s and every other link at 8 GT/s, all
  card to card, 64 KiB-4 MiB x 3,000, f32:

  | Orderings | Wrong elements |
  |---|---|
  | four cards, `readback` on the 2060's two pairs, `release` on the other two | 175,296 / 90,144 / 114,204 (3 runs) |
  | four cards, `readback` on every pair | **0 / 0 / 0** |
  | RTX 3090 + V100 (both at 8 GT/s), `release`, while another program sends through the slow port | **25,376** |
  | the same RTX 3090 + V100, `release`, alone | 0 |
  | RTX 2060 + RTX 3090 (the slow pair), `release` / `readback` | 10 of 10 short runs wrong / 0 |

  A program cannot see what other programs send through the machine, so no check inside the fork can make `release`
  safe. Use it only where every card's link runs at 8 GT/s and x4 or wider and nothing sends through a slower one; with
  `release` set on a pair whose own link is slow, the fork warns.
- **What `readback` costs**: little. RTX 3090 + V100 on CPU lanes (x4 each), Qwen3.8-27B Q8_0, prompt 2048, tokens/s,
  2 rounds: stock NCCL 819-831, the fork with `release` 861-862, with `readback` 856-857 (0.5 % less); generation 32.2
  with every route; `all_reduce_perf` the same for both orderings (2.07-2.08 GB/s at 128 MiB). The 4 bytes read back
  per step are far from what stopped a V100 behind a chipset (bursts of 1 MiB:
  [troubleshooting](troubleshooting.md#a-tesla-v100-in-a-chipset-slot-freezes-xid-62-while-another-card-reads-its-memory)).
- **Links between bursts**: a GeForce card runs its link at 2.5 GT/s when idle. With a program running, the links in use
  stayed at 8 GT/s through 3-second pauses (sampled four times a second), and `release` held across them (0 wrong): an
  idle card's slow link only matters once it carries traffic.

## Faster on CPU lanes: both routes at once

Where every card has its own CPU link, the card-to-card route alone was slower than stock NCCL on three or more cards
(above). Since 2026-11-27 the fork can send part of the large lane's channels card to card and the rest through host
memory: `NCCL_DRMSHARE_CHANNEL_MIX=<d>:<s>` puts `d` of every `d+s` channels on DRMShare and `s` on NCCL's shared
memory. The same four cards as above (GTX 1070 Ti, RTX 2060, RTX 3090, V100, PCIe 3.0 x4 each), every result checked:

- **The card-to-card route is bandwidth-bound here, not latency-bound**: NCCL's buffer size (4-16 MiB, so fewer copies
  and fewer start costs) and channel count (1-4) changed nothing (four cards 1.42-1.45 GB/s, two 2.06-2.11).
- `all_reduce_perf` at 128 MiB, GB/s (busbw):

  | | four cards | RTX 3090 + V100 |
  |---|---|---|
  | stock NCCL | 2.10 | 2.07-2.14 |
  | stock NCCL, `NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4` | 2.14 | 2.21 |
  | the fork, card to card only | 1.43 | 2.08 |
  | the fork, mix `1:1` (two channels) | 1.92 | 2.17 |
  | **the fork, mix `1:3` with four channels** | **2.15** | **2.24** |
  | the fork, mix `3:1` with four channels | 1.64 | 2.06 |

- llama.cpp-drmshare, Qwen3.8-27B Q8_0, `-sm tensor -ub 2048`, prompt 2048 tokens/s (2 rounds); generation was the same
  on every route (20.1-20.6 on four power-capped cards, 32.1-32.3 on the pair):

  | | four cards (`-ts 4/3/16/23`, power-capped) | RTX 3090 + V100 (`-ts 12/15`) |
  |---|---|---|
  | stock NCCL | 621-624 | 821-831 |
  | stock NCCL, four channels | 644 | 824 |
  | the fork, card to card only | 544-545 | 856-861 |
  | **the fork, mix `1:3`, four channels** | **657-658 (+6 % over stock)** | **869-872 (+5 %)** |

- **Correct**: the back-to-back test through the mix on four cards, 4-32 MiB x 300, f32 and bf16, 0 wrong; perplexity
  identical chunk for chunk to stock NCCL **with four channels** (6.0012; with two channels both give 5.9937: four
  channels sum in another order, in stock NCCL too).

So on such a machine use

```bash
export NCCL_DRMSHARE_CHANNEL_MIX=1:3 NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4
```

It is an opt-in because the best ratio depends on the machine: where a card's own link to host memory is the narrow
one (the V100 in a chipset slot at the top of this page), the card-to-card route alone was far faster than host
memory, and an even split by channel would hold it back. The mix was not tried there.

## How it works

- **The transport** (`src/transport/drmshare.cc`) follows NCCL's P2P transport in its copy-engine mode. The receiving
  card's buffer is Vulkan memory exported twice: as a dma-buf, which the sending card's Vulkan imports through
  DRMShare, and as a file descriptor that the receiving card's CUDA maps for NCCL's kernel. The sending kernel writes
  each step into a local buffer; NCCL's proxy thread copies the step across with a copy engine, waits for its fence,
  and then raises the "data ready" mark that the receiving kernel polls.
- **The pair check** (`drmsharePair`, from NCCL's `canConnect`, once per pair): each card writes 4 MiB into the other's
  memory through DRMShare and reads the last 16 bytes back, and the receiving card copies its own memory out and
  compares every word. A pair that fails is refused; on the large lane NCCL then tries its shared-memory transport,
  which comes after DRMShare there. Both ends of a connection get the same answer. The links (each card's PCIe path to
  its root port in sysfs) are read for the log, and for a warning when `release` is set on a slow one.
- **Ordering.** `readback` (default): after each copy, 4 bytes are read back from the destination (a PCIe read cannot
  pass the writes before it); a card in `NCCL_DRMSHARE_NOREAD` fetches the data itself instead, so it is never read
  across. `release` (opt-in): every card pushes, and each copy ends with a Vulkan queue-family release to the other
  device. Each card has two copy queues, so pushing and fetching run side by side.
- **Simple protocol only.** NCCL's LL and LL128 protocols put their flags inside the data, which needs writes to land
  in order; a copy engine does not promise that. The small lane, which uses LL, stays on NCCL's shared memory.
- **Two lanes** (`NCCL_DRMSHARE_SPLIT_BYTES`): each communicator gets a companion over the same GPUs that may use only
  NCCL's P2P and shared-memory transports; `ncclAllReduce` sends a collective below the cut to the companion.
- **No CUDA call in the proxy's copy loop**: a CUDA call there can wait on a driver lock held by the program's own
  thread while it loads a CUDA module, while NCCL's kernel waits for the proxy: a hang.
- **Why not real P2P?** CUDA cannot be given a pointer into the other card's memory on this pair: no peer access
  between two generations, CUDA's dma-buf import exists only on Tegra Thor, and NVIDIA's closed 580 driver refuses to
  register another card's memory window as I/O memory. Vulkan, through DRMShare, can reach it: hence copy engines.

## Limits

- **One process only.** Programs with one process per GPU (PyTorch's DDP, vLLM, most NCCL launchers) get no `via DRM`
  connection.
- **Tested on**: a Tesla V100 + RTX 3090 and a GTX 750 Ti + RTX 3090 on NVIDIA's closed 580 (where `release` failed on
  the Maxwell card and `readback` held: [Maxwell](#maxwell-a-gtx-750-ti-with-an-rtx-3090)), a V100 + GTX 1650 SUPER, and
  rings over three, four and five generations on 580 (Maxwell, Pascal, Volta, Turing, Ampere: [three-generations.md](three-generations.md)),
  and three RTX 3090s on driver 615 for four hours ([above](#three-rtx-3090s-on-driver-615-a-four-hour-stress)). The
  pair check and the `readback` default on four of those cards ([safe by default](#safe-by-default-every-pair-checked-readback-everywhere)).
- **Not always faster**: on cards that all have their own CPU lanes, stock NCCL through host memory was faster than the
  card-to-card route alone ([three generations](#three-generations-a-v100-a-gtx-1650-super-and-an-rtx-3090)); there the
  channel mix makes the fork a few per cent faster than stock ([both routes at once](#faster-on-cpu-lanes-both-routes-at-once)),
  and the large gains come where a card's link to host memory is the narrow one.
- **A pair that cannot carry peer traffic goes through host memory.** Two cards behind one AMD or Intel chipset cannot
  reach each other ([troubleshooting](troubleshooting.md#two-cards-behind-one-amd-chipset-the-copy-finishes-nothing-arrives));
  the pair check finds that and sends that pair through NCCL's shared memory on the large lane, the others card to
  card. Tested with one pair refused on purpose and with every import failing; two real cards behind one chipset were
  not in the machine for this test. With every size card to card and `NCCL_SHM_DISABLE=1`, NCCL stops with "No
  transport found" for such a pair.
- **Outside the Vulkan specification**: NVIDIA's Vulkan driver lists a dma-buf and an opaque file descriptor as not
  compatible for one allocation; the transport exports one allocation as both, which works on 580 (every result
  checked) but could stop working in another driver version.
- **Copy-engine latency**: a copy across takes 28-45 µs to start and finish, so transfers below about 2 MiB are faster
  through host RAM; that is why the two lanes exist.
- The rest of DRMShare's limits apply: the board must carry peer traffic, and a V100 in a chipset slot must not be read
  across in large bursts (the default `readback` reads 4 bytes per step; `NCCL_DRMSHARE_NOREAD` avoids even that).
