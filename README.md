# DRMShare: share video memory between NVIDIA GPUs of different generations

*Formerly Hidden GEM. DRM as in Linux's Direct Rendering Manager, the kernel's graphics subsystem it is built on.*

One opt-in option in `nvidia-drm`, the part of NVIDIA's Linux driver that ships as source, lets **Vulkan** on one GPU
import another GPU's video memory through a Linux **dma-buf**. Each card then copies straight into the other's memory
over PCIe, instead of through host RAM. It works on pairs CUDA refuses, because they are of different generations: a
Tesla V100 with an RTX 3090, an RTX 2060, a GTX 1650 SUPER or a GTX 1070 Ti, a GTX 750 Ti (Maxwell) with an RTX 3090,
and an RTX 2060 with an RTX 3060, and up to four generations in one machine at once, every pair sharing with every
other ([docs/three-generations.md](docs/three-generations.md)). It runs on NVIDIA's own
drivers 580, 595 and 615, without a patched driver and without a patched `libcuda`. Vulkan programs use it directly,
and CUDA programs that exchange data through **NCCL**, such as llama.cpp's CUDA backend, use it through an
experimental NCCL fork, a drop-in `libnccl.so.2` ([docs/nccl.md](docs/nccl.md)).

The name: the sharing comes from `nvidia-drm`, NVIDIA's part of Linux's **DRM**, and the buffers it shares are DRM's
GEM objects; the ability was already there, hidden in NVIDIA's own open-source code (hence the old name). The change
adds 110 lines (580) or 144 lines (615) to three `nvidia-drm` files, and the option is off by default.

**This is memory sharing, not CUDA P2P.** CUDA itself, PyTorch and vLLM see no peer access, as before: only Vulkan
programs, and NCCL through the fork, use the shared memory. llama.cpp can use it both ways: its CUDA backend built with
NCCL through the fork, with no change to llama.cpp, and its Vulkan backend with a second small change, where its
tensor-parallel mode adds up each layer's results through the shared memory
([what this is, and what it is not](docs/volta-dmabuf.md#what-this-is-and-what-it-is-not)).

**Use CUDA P2P where you can.** If your pair has a P2P route in the companion guide,
[how-to-enable-p2p-on-nvidia-gpus](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus), use that: it works for every CUDA program (NCCL, PyTorch, vLLM). For llama.cpp, its CUDA backend built with NCCL
was faster than the Vulkan backend with memory sharing even without P2P, on both setups where both were measured: three
RTX 3090s ([D5](docs/dmabuf-615.md#d5-three-rtx-3090s)) and the Tesla V100 + RTX 3090, where CUDA has no peer access at
all (719 against 412 tokens/s on prompts, [V6](docs/volta-dmabuf.md#v6-llamacpp-tensor-parallel-with-an-all-reduce-through-shared-video-memory)); there the NCCL fork
brings memory sharing to the CUDA backend too (932 tokens/s, [docs/nccl.md](docs/nccl.md)).
DRMShare is for the cases where P2P is not possible or does not fit:

- a **Tesla V100 or a Pascal card with a card of another generation**: these cards run only on NVIDIA's closed driver
  580 (the guide's patched driver cannot run them), and CUDA refuses peer access between two architectures;
- **two generations of Turing or newer on NVIDIA's stock driver** (the guide's routes for them need its patched driver,
  plus a patched copy of `libcuda` or root);
- **programs that use Vulkan rather than CUDA**, like llama.cpp's Vulkan backend.

The guide keeps the same list: [When P2P is not possible](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus#when-p2p-is-not-possible-drmshare).

**Experimental.** Tested on three machines, three driver versions and eight setups (seven pairs, and three RTX 3090s
together), every word of every copy checked: about 21 hours of tests, plus a four-hour stress of the NCCL fork on three
RTX 3090s with every answer identical to stock NCCL ([nccl.md](docs/nccl.md#three-rtx-3090s-on-driver-615-a-four-hour-stress)).
Two known limits: a Tesla V100 in a chipset slot can freeze when another card reads its memory while a program starts or
stops on it ([troubleshooting](docs/troubleshooting.md#a-tesla-v100-in-a-chipset-slot-freezes-xid-62-while-another-card-reads-its-memory)),
and two cards behind the same AMD chipset cannot share memory at all: nothing arrives and nothing says so
([troubleshooting](docs/troubleshooting.md#two-cards-behind-one-amd-chipset-the-copy-finishes-nothing-arrives)).
The changed module is loaded by hand and gone after a reboot.

**Problems and field reports** about DRMShare, the NCCL fork (its repository has no issue tracker) or llama.cpp-drmshare:
[issues here](https://github.com/Cinderella-Man/drmshare/issues). About CUDA P2P (the patched driver, the patched copy
of `libcuda`): [the P2P guide's issues](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/issues).

## Which page is yours

| Your cards | Driver | Follow | Tested |
|---|---|---|---|
| Tesla V100 + a GeForce card (Volta + Ampere, Turing or Pascal), or two V100s; a Maxwell or Pascal card with a newer one | NVIDIA's driver 580 (the closed package: the open modules do not run Volta, Pascal or Maxwell) | [docs/volta-dmabuf.md](docs/volta-dmabuf.md) | **works**: V100 + RTX 3090, V100 + RTX 2060, V100 + GTX 1070 Ti, V100 + GTX 1650 SUPER ([three cards](docs/volta-dmabuf.md#v100--gtx-1650-super-volta--turing-tu116-three-cards-in-one-machine)); two V100s; GTX 750 Ti + RTX 3090 ([Maxwell](docs/volta-dmabuf.md#gtx-750-ti--rtx-3090-maxwell--ampere-no-v100)) |
| Two or more cards of Turing or newer, any generations | Ubuntu's driver 615 (or 595) | [docs/dmabuf-615.md](docs/dmabuf-615.md) | **works**: RTX 2060 + RTX 3060; three RTX 3090s ([D5](docs/dmabuf-615.md#d5-three-rtx-3090s)) |
| Another driver version (580 open modules, 590, 610, 620) | | [docs/dmabuf-drivers.md](docs/dmabuf-drivers.md) | 580 open modules **works**; 590, 610, 620: the change applies, not run |
| NCCL between two cards CUDA cannot pair (llama.cpp's CUDA backend, other one-process NCCL programs) | 580 (V100, Maxwell) or 615 | [docs/nccl.md](docs/nccl.md) | **experimental**: V100 + RTX 3090; large all-reduces +44 %, llama.cpp +29 % on prompts (+5 % on top of llama.cpp's own speed-ups); GTX 750 Ti + RTX 3090 ([Maxwell](docs/nccl.md#maxwell-a-gtx-750-ti-with-an-rtx-3090)); V100 + GTX 1650 SUPER, and one ring over three, four and five generations (Maxwell, Pascal, Volta, Turing, Ampere), correct but slower than stock NCCL when every card has CPU lanes ([three generations](docs/nccl.md#three-generations-a-v100-a-gtx-1650-super-and-an-rtx-3090)); three RTX 3090s on 615, a four-hour stress with every answer identical to stock NCCL, prompts +22 % ([stress](docs/nccl.md#three-rtx-3090s-on-driver-615-a-four-hour-stress)); since 2026-11-27 every pair checked before use (a pair that fails goes through host memory) and the safe `readback` ordering by default ([safe by default](docs/nccl.md#safe-by-default-every-pair-checked-readback-everywhere)); on cards with their own CPU links, part of the traffic through host memory makes it 5-6 % faster than stock NCCL ([both routes](docs/nccl.md#faster-on-cpu-lanes-both-routes-at-once)) |
| One model across both cards with llama.cpp | any of the above | [benchmarks/llama](benchmarks/llama/README.md), [volta-dmabuf.md V6](docs/volta-dmabuf.md#v6-llamacpp-tensor-parallel-with-an-all-reduce-through-shared-video-memory) | **works**: Qwen3.8-27B on V100 + RTX 3090, on RTX 2060 + RTX 3060 and on three RTX 3090s |

Four things decide whether it works on your machine:

- **The board must carry peer traffic**, as for CUDA P2P. On a mining board with PCIe switches nothing arrived, and
  nothing said so ([volta-dmabuf.md](docs/volta-dmabuf.md#on-a-board-that-does-not-carry-peer-traffic); what is known
  per board: the P2P guide's [boards page](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/blob/main/docs/boards.md)).
  The same between two cards behind one AMD chipset: keep at most one card there, the others on CPU lanes
  ([troubleshooting](docs/troubleshooting.md#two-cards-behind-one-amd-chipset-the-copy-finishes-nothing-arrives)).
- **A card shares only what fits its memory window** (BAR1). RTX 30 cards and newer show all their memory with Resizable
  BAR on; a Turing card's stock 256 MB window shares buffers up to 224 MiB, and the P2P guide's
  [turing.md](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/blob/main/docs/turing.md) T1-T3
  enlarge it. Pascal and Maxwell cards (a GTX 1070 Ti, a GTX 750 Ti) keep a fixed 256 MB window: buffers of up to about
  192-224 MiB.
- **Programs that map a lot of video memory** (llama.cpp with Resizable BAR) need the driver's static BAR1 off,
  `RMForceStaticBar1=0`, for every card ([troubleshooting](docs/troubleshooting.md)).
- **Where a Tesla V100 sits, if other cards read its memory.** In a slot wired through the board's chipset, a V100
  whose memory another card reads in large pieces can freeze when a program starts or stops on it. In a CPU slot it
  passed every run, and writing into it is safe in either slot: that is all llama.cpp does
  ([troubleshooting](docs/troubleshooting.md#a-tesla-v100-in-a-chipset-slot-freezes-xid-62-while-another-card-reads-its-memory)).

## Results

Copies of 1 GiB into the first card's memory / into the second card's, every word checked, against CUDA's
`cudaMemcpyPeer`, which goes through host RAM for every one of these pairs:

| Driver | Cards | Memory sharing | Both ways at once | 4-byte copy | Through host RAM |
|---|---|---|---|---|---|
| 580, closed | Tesla V100 (PCIe 3.0 x4 chipset slot) + RTX 3090 (PCIe 4.0 x16) | **2.96 / 3.26 GB/s** | **5.62 GB/s** | **2.4 / 3.1 µs** | 2.50 / 2.79 GB/s, 17.0 / 19.5 µs |
| 580, closed | Tesla V100 (PCIe 3.0 x4 chipset slot) + RTX 2060 (PCIe 3.0 x16) | **2.91 / 3.28 GB/s** | **5.62 GB/s** | **2.2 / 3.2 µs** | 2.27 / 2.51 GB/s, 19.9 / 21.5 µs |
| 580, closed | Tesla V100 (PCIe 3.0 x4 chipset slot) + GTX 1070 Ti (PCIe 3.0 x16) | **2.96 / 3.28 GB/s** | **5.63 GB/s** | **1.5 / 2.2 µs** | 2.82 / 3.22 GB/s, 18.4 / 20.4 µs |
| 580, closed | GTX 750 Ti (PCIe 3.0 x4 chipset slot) + RTX 3090 (PCIe 4.0 x16) | **2.95 / 3.28 GB/s** | **5.63 GB/s** | **2.4 / 1.9 µs** | 2.52 / 2.80 GB/s, 14.4 / 18.1 µs |
| 580 open, 595, 615 | RTX 2060 (PCIe 3.0 x4 chipset slot) + RTX 3060 (PCIe 4.0 x16) | **2.93-2.97 / 3.18-3.28 GB/s** | **5.61-5.64 GB/s** | **1.6-2.2 / 2.7-3.1 µs** | 2.47 / 2.78 GB/s, 13.5 / 17.8 µs |
| 615 open | RTX 3090 (PCIe 4.0 x4 chipset slot) + RTX 3090 (PCIe 4.0 x8) | **6.60 / 6.60 GB/s** | **12.84 GB/s** | **3.2 / 2.6 µs** | 4.11 / 4.02 GB/s, 18.8 / 15.5 µs |

**The slot sets these speeds, not the method:** a PCIe 3.0 x4 link carries about 3.3 GB/s each way at most, and every
route between two cards, memory sharing, host RAM or CUDA P2P, is capped by the slower card's link. Both ways at once
is 5.6 GB/s for every pair on the first board: its PCIe 3.0 x4 chipset slot is the ceiling. Two V100s on another
board reach 1.67 / 1.63 GB/s in a PCIe 2.0 x4 slot, the same as CUDA P2P there. On a third board, the two RTX 3090s on
x8 CPU lanes copy at 13.1-13.2 GB/s ([D5](docs/dmabuf-615.md#d5-three-rtx-3090s)).

llama.cpp, tensor parallel, with the all-reduce through shared video memory instead of host RAM:

| Cards, model | Results | Prompt | Generation |
|---|---|---|---|
| V100 (PCIe 3.0 x4 chipset slot) + RTX 3090 (PCIe 4.0 x16), Qwen3.8-27B Q8_0 | identical (perplexity equal on every chunk) | **+19 %** (413 vs 347 tokens/s) | +4 % (28.9 vs 27.8) |
| the same cards with the slots swapped: V100 (PCIe 3.0 x16) + RTX 3090 (PCIe 3.0 x4 chipset slot) | identical | **-18 %** (388 vs 474) | 0 % (30.2 vs 30.1) |
| three RTX 3090s (one at PCIe 4.0 x4 behind the chipset, two at PCIe 4.0 x8), Qwen3.8-27B Q8_0, driver 615 | identical (perplexity equal on every chunk) | **+45 %** (1213 vs 835) | the same (37.0 vs 37.2) |
| RTX 2060 (PCIe 3.0 x4 chipset slot) + RTX 3060 (PCIe 4.0 x16), Qwen3.8-27B UD-Q3_K_XL, drivers 580 / 595 / 615 | identical on each driver | -1.5 to -3.5 % | +0.7 to +1.1 % (19.3 vs 19.2) |

Prompts of 2048 tokens; the two routes alternated, same binary. **On the V100 + RTX 3090 the slot layout decides.** The
V100 is the slower card and sets the pace. With it on the narrow x4 link, the shared all-reduce saves it from reading
host RAM over that link; with it on x16, host RAM is the better route and stock llama.cpp is faster
([volta-dmabuf.md V6](docs/volta-dmabuf.md#v6-llamacpp-tensor-parallel-with-an-all-reduce-through-shared-video-memory)).
On three RTX 3090s the gain shrinks with longer prompts: +40 % at 8,000 tokens, a few percent from 32,000 to 262,000
(single runs), with the generated text identical at every length ([D5](docs/dmabuf-615.md#d5-three-rtx-3090s)).

**NCCL, through the experimental fork** ([docs/nccl.md](docs/nccl.md)), on the same V100 (PCIe 3.0 x4 chipset slot) +
RTX 3090 (PCIe 4.0 x16), NCCL's `all_reduce_perf` with every result checked:

| NCCL all-reduce, two cards | 8 B | 2 MiB | 128 MiB |
|---|---|---|---|
| NCCL as shipped (through host RAM) | 13.6 µs | 1.75 GB/s | 1.82 GB/s |
| **the fork (through the shared memory)** | 13.0 µs | **2.18 GB/s** | **2.63 GB/s (+44 %)** |

The ceiling is the V100's x4 slot with both directions busy, about 2.8 GB/s. Small all-reduces take NCCL's own route
through host RAM either way.

**llama.cpp's CUDA backend is faster than its Vulkan backend on this pair, and the fork speeds it up.** With V6's
llama.cpp commit built for CUDA 12.4 with NCCL 2.32.6, every transfer through host memory because CUDA refuses peer
access between two architectures: 719-720 tokens/s on 2048-token prompts and 33.2 generating with NCCL, 647 / 34.7 with
its built-in all-reduce, against 412 / 28.9 for the Vulkan backend with memory sharing. The NCCL fork moves the CUDA
backend's large all-reduces through the shared memory: 932 tokens/s (+29 %), with no change to llama.cpp. llama.cpp's
own tensor-parallel changes in [llama.cpp-drmshare](https://github.com/Cinderella-Man/llama.cpp-drmshare) help more, with stock NCCL as well (974-984,
the P2P guide's [llama.cpp section](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus#more-speed-tensor-parallel-changes-in-a-llamacpp-copy-experimental)),
and the fork adds 5 % on top of them (1,017-1,028). Perplexity was identical in every comparison. On three RTX 3090s
the CUDA backend with NCCL was faster than Vulkan too ([D5](docs/dmabuf-615.md#d5-three-rtx-3090s)); the fork has not
run there yet.

Every llama.cpp route on the V100 + RTX 3090 in one table (rows 5 and 7 use the NCCL fork, rows 6-8 llama.cpp-drmshare's
changes; row 7 is the fastest on prompts, row 8 on generation):

![Tesla V100 + RTX 3090, Qwen3.8-27B Q8_0, llama.cpp tokens/s, 2048-token prompt / generation: 1 Vulkan through host RAM 347 / 27.9; 2 Vulkan + DRMShare 412 / 28.9; 3 CUDA, llama.cpp's own all-reduce 647 / 34.7; 4 CUDA + NCCL through host RAM 720 / 33.2; 5 CUDA + the NCCL fork (DRMShare transport) 932 / 33.1; 6 CUDA + NCCL + llama.cpp-drmshare's speed-ups, -ts 3/1, 974-984 / 33.8; 7 the same with the NCCL fork 1,017-1,028 / 33.7; 8 the speed-ups at -ts 3/2 925 / 37.3](docs/assets/llama-routes-v100-3090.png)

So for llama.cpp, use the CUDA backend built with NCCL, with the fork on pairs CUDA cannot connect; memory sharing also
speeds up the Vulkan backend and other Vulkan programs ([V6](docs/volta-dmabuf.md#v6-llamacpp-tensor-parallel-with-an-all-reduce-through-shared-video-memory)).

On the RTX 2060 + RTX 3060 the shared all-reduce gives no meaningful gain; tensor parallel itself generates 42 % faster
than layer split there ([benchmarks/llama](benchmarks/llama/README.md)).

## Every pair so far

The goal is for DRMShare to work on every pair, including those where CUDA P2P already works, so this table lists
every pair from this repository and from the P2P guide, side by side. The `not tested` cells are the to-do list.

| Cards | Generations | CUDA P2P | DRMShare |
|---|---|---|---|
| Tesla V100 + RTX 3090 | Volta + Ampere | **not possible**: the V100 needs the closed driver 580, and CUDA refuses peer access between two architectures (NVIDIA's `ForceP2P` key makes the kernel driver agree, but CUDA still offers no peer access, also with a patched copy of `libcuda` for 580; allowing P2P reads as well froze the machine with the V100 in a chipset slot: [volta-dmabuf.md](docs/volta-dmabuf.md#nvidias-forcep2p-key-the-kernel-says-yes-cuda-still-says-no)) | **works** on 580: 2.96 / 3.26 GB/s, 5.62 GB/s both ways, soak 20/20; llama.cpp Qwen3.8-27B identical, prompts +19 % with the V100 in the x4 chipset slot (with it in x16, host RAM is faster), while llama.cpp's CUDA backend is faster than both even through host RAM (719 against 412 tokens/s) ([volta-dmabuf.md](docs/volta-dmabuf.md)); NCCL through the experimental fork: large all-reduces +44 %, llama.cpp's CUDA backend +29 % on prompts, 1,017-1,028 tokens/s with llama.cpp-drmshare's changes ([nccl.md](docs/nccl.md)); a V100 in a chipset slot can freeze when read while a program starts or stops on it ([troubleshooting](docs/troubleshooting.md#a-tesla-v100-in-a-chipset-slot-freezes-xid-62-while-another-card-reads-its-memory)) |
| Tesla V100 + RTX 2060 | Volta + Turing | **not possible**, as above | **works** on 580: 2.91 / 3.28 GB/s, soak 20/20 (with an 8 GB window on the RTX 2060; with its stock 256 MB window, buffers up to 224 MiB) |
| Tesla V100 + GTX 1070 Ti | Volta + Pascal | **not possible**, as above | **works** on 580: 2.96 / 3.28 GB/s, soak 20/20; the 1070 Ti shares up to 224 MiB (256 MB window) |
| Tesla V100 + GTX 1650 SUPER | Volta + Turing (TU116) | **not possible**, as above | **works** on 580 through a USB-cable x1 riser held at 2.5 GT/s: 0.21 / 0.21 GB/s (the riser's limit), 0 wrong, also with a V100 + RTX 3090 pair busy at the same time; NCCL through the fork with `readback` (the default `release` let wrong data through), llama.cpp answers identical to stock NCCL ([volta-dmabuf.md](docs/volta-dmabuf.md#v100--gtx-1650-super-volta--turing-tu116-three-cards-in-one-machine), [nccl.md](docs/nccl.md#three-generations-a-v100-a-gtx-1650-super-and-an-rtx-3090)) |
| GTX 750 Ti + RTX 3090 | Maxwell + Ampere | **not possible** on 580: with NVIDIA's `ForceP2P` key allowing reads and a patched `libcuda`, CUDA says it can, but enabling peer access fails in the driver both ways (no freeze with this card) ([volta-dmabuf.md](docs/volta-dmabuf.md#nvidias-forcep2p-key-the-kernel-says-yes-cuda-still-says-no)) | **works** on 580: 2.95 / 3.28 GB/s, 5.63 GB/s both ways, soak 20/20; the 750 Ti shares up to about 192 MiB (256 MB window); llama.cpp Vulkan perplexity identical; NCCL through the fork with `readback` ordering (the default `release` let wrong data through twice in about 50 runs) ([volta-dmabuf.md](docs/volta-dmabuf.md#gtx-750-ti--rtx-3090-maxwell--ampere-no-v100), [nccl.md](docs/nccl.md#maxwell-a-gtx-750-ti-with-an-rtx-3090)) |
| Two Tesla V100s | Volta | **works** on the stock 580 with the IOMMU in passthrough ([P2P guide](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/blob/main/docs/pascal-and-volta.md)) | **works** on 580: 1.67 / 1.63 GB/s, the same as CUDA P2P in that PCIe 2.0 x4 slot |
| RTX 2060 + RTX 3060 | Turing + Ampere | **works with workarounds**: the P2P guide's patched driver plus a patched copy of `libcuda`, or writes through the memory window as root ([P2P guide](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/blob/main/docs/mixed-generations.md)) | **works** on 580 (open modules), 595 and 615: 2.93-2.97 / 3.18-3.28 GB/s, 5.6 GB/s both ways, soaks 10/10 and 20/20; llama.cpp Qwen3.8-27B identical, same speed ([benchmarks/llama](benchmarks/llama/README.md)) |
| Three RTX 3090s | Ampere | **works** with the patched driver 615 | **works** on 615 (open modules): 6.60 GB/s through a x4 chipset link, 13.1-13.2 GB/s between two x8 cards, all six directions at once for over an hour, 56 TB checked, 0 wrong; llama.cpp Qwen3.8-27B identical, prompts +45 % ([dmabuf-615.md D5](docs/dmabuf-615.md#d5-three-rtx-3090s)); the NCCL fork on all three for four hours, every answer identical to stock NCCL, 6.37 trillion all-reduce elements checked, 0 wrong ([nccl.md](docs/nccl.md#three-rtx-3090s-on-driver-615-a-four-hour-stress)) |
| RTX 3080 12 GB + RTX 3080 10 GB | Ampere (one chip) | **works** with the patched driver 615 | not tested |
| RTX 2060 + GTX 1650 SUPER | Turing (two chips) | **works** with the patched driver and a chip-ID shim for CUDA | not tested |
| Two P106-090s, in the test rig | Pascal | **works** on the stock 580 | not tested |
| GTX 1070 Ti + RTX 2060 | Pascal + Turing | **not possible**: two architectures, and Pascal needs 580 | not tested (the V100 works with each of them) |
| RTX 3090 + GTX 1650 SUPER, both behind one AMD B550 chipset | Ampere + Turing | not tested here; on an Intel Z390, CUDA P2P between two chipset cards lost every write (outside report, [P2P guide issue #5](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/issues/5)) | **does not work**: every word lost both ways, no error; the chipset answers with "Unsupported Request"; each card works with a card on CPU lanes ([troubleshooting](docs/troubleshooting.md#two-cards-behind-one-amd-chipset-the-copy-finishes-nothing-arrives)). **The same two cards on CPU lanes** (a bifurcation splitter): 3.30 GB/s both ways, 0 wrong, and the NCCL fork's ring over all three generations runs ([volta-dmabuf.md](docs/volta-dmabuf.md#v100--gtx-1650-super-volta--turing-tu116-three-cards-in-one-machine)) |
| P106-090s behind the PCIe switches of a mining board | Pascal | **does not work**: the board drops the traffic | **does not work**: nothing arrives, no error ([volta-dmabuf.md](docs/volta-dmabuf.md#on-a-board-that-does-not-carry-peer-traffic)) |
| Ada, Hopper, Blackwell cards | | not tested | not tested (595 and 615 run Ada, Hopper and Blackwell on the same driver code as the cards above) |

## Contents

| | |
|---|---|
| [docs/volta-dmabuf.md](docs/volta-dmabuf.md) | Where it started: a Tesla V100 with an RTX 3090 (also an RTX 2060 and a GTX 1070 Ti) on NVIDIA's driver 580. Why nothing else works for a V100, the change, how to build and load it, tests, the llama.cpp all-reduce (V6), a board that does not carry peer traffic. |
| [docs/dmabuf-615.md](docs/dmabuf-615.md) | The same on Ubuntu's driver 615 for Turing and newer, tested on an RTX 2060 + RTX 3060; llama.cpp on that pair (D4); three RTX 3090s at once (D5). |
| [docs/dmabuf-drivers.md](docs/dmabuf-drivers.md) | Every NVIDIA branch from 580 to 620: which version of the change it takes, what was tested, all results in one table, llama.cpp on each driver, how to build for another version. |
| [docs/three-generations.md](docs/three-generations.md) | Mixed generations in one machine on a bifurcation splitter: three sets of three (Turing, Pascal or Maxwell + Ampere + Volta) and four at once (Turing + Maxwell + Ampere + Volta): memory sharing on every pair (about 200 GB checked per set, 0 wrong), the NCCL fork's ring over all of them, Qwen3.8-27B across them with answers identical to stock NCCL; `release` ordering needs a fast link. |
| [docs/nccl.md](docs/nccl.md) | **Experimental**: an NCCL fork with a transport through the shared memory, for CUDA programs on pairs CUDA cannot connect: build, settings, checks, results on the V100 + RTX 3090 (llama.cpp +29 % on prompts, +5 % on top of llama.cpp's own speed-ups; the slots swapped), a GTX 750 Ti, three RTX 3090s on 615 (a four-hour stress) and three generations at once; when to use `readback`. |
| [docs/troubleshooting.md](docs/troubleshooting.md) | A Tesla V100 in a chipset slot that freezes while another card reads its memory; two cards behind one chipset that cannot reach each other; a card on an x1 riser that falls off the bus; the kernel log filling with `dmaAllocMapping_GM107` lines (static BAR1); driver 595's per-card option that misses one card, with a fix. |
| [docs/test-rig-log.md](docs/test-rig-log.md) | What was run on the test rig on 2026-11-25, in order: what worked, what did not, the dead ends and false alarms, and what the tests do not show. |
| [benchmarks/llama](benchmarks/llama/README.md) | llama.cpp tensor parallel on the RTX 2060 + RTX 3060: scripts, results on three drivers, and a crash at exit on driver 615. |
| [tools/](tools) | `p2p_dmabuf.cpp` (the Vulkan test: share, copy both ways, check every word), `staged_baseline.cu` (CUDA through host RAM, the baseline), `keep_busy.cu` (keeps a receiving card out of its idle state), `bar1_probe.cpp` (how much host-visible video memory Vulkan can map), `p2p_iomem.cu` (the P2P guide's I/O-memory route, compared in volta-dmabuf.md), `stale_test.cu` (thousands of NCCL all-reduces back to back, every element checked: [nccl.md](docs/nccl.md)); the changed `nvidia-drm` for 580 and 615, the changed llama.cpp and the NCCL fork as pinned submodules. |
| [scripts/](scripts) | The steps as scripts, for a test machine: `drv-test.sh` (build DRMShare's `nvidia-drm` for the running driver and run every test: D1-D3 in one go), `switch-driver.sh` (another Ubuntu driver branch), `run-driver.sh` (NVIDIA's `.run` installer for branches Ubuntu does not package), `reload-nvidia.sh` (reload with driver keys, without a reboot), `x1-riser-gen1.sh` with its service (keeps cards on x1 risers at a link speed the riser carries), `dflash_add_head.py` (gives a DFlash draft its own embeddings and head, for llama.cpp's tensor split). |
| [patches/](patches) | `595-per-device-registry-keys.diff`: 615's per-card option lookup for driver 595 ([troubleshooting](docs/troubleshooting.md)). |
| [docs/assets](docs/assets) | Infographics: V100 + RTX 3090, and five generations on driver 580 (`make_infographic.py`); every llama.cpp route on the V100 + RTX 3090 (`llama-routes-v100-3090.html`, the table above), and in both slot layouts (`llama-routes-v100-3090-swapped.png`). |

The code lives in three forks, pinned here as submodules:
[Cinderella-Man/open-gpu-kernel-modules](https://github.com/Cinderella-Man/open-gpu-kernel-modules), branches
`580-peer-vidmem-dmabuf` and `615-peer-vidmem-dmabuf` (the `nvidia-drm` option `peer_vidmem_dmabuf`), and
[Cinderella-Man/llama.cpp-drmshare](https://github.com/Cinderella-Man/llama.cpp-drmshare), branch `vk-allreduce-shared-vram` (llama.cpp
with `GGML_VK_ALLREDUCE_P2P=1`), and [Cinderella-Man/nccl-drmshare](https://github.com/Cinderella-Man/nccl-drmshare),
branch `2.32.6-drmshare` (NCCL 2.32.6 with the transport of [docs/nccl.md](docs/nccl.md)). Code, option and log names still say P2P after the hardware path; the text says memory
sharing.

These pages started in the P2P guide and moved here on 2026-11-25, with their history.

## Credits

- Christian König and the [amdgpu](https://sources.debian.org/src/linux/6.19.6-2/drivers/gpu/drm/amd/amdgpu/amdgpu_dma_buf.c)
  developers: the dma-buf technique for sharing video memory between devices.
- [NVIDIA open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules): `nvidia-drm` and the driver's
  kernel interface as source, dual MIT/GPLv2.
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp): tensor parallelism, and the Vulkan all-reduce by pwilkin
  ([PR #25051](https://github.com/ggml-org/llama.cpp/pull/25051)) that the shared-memory all-reduce builds on.
- [TheBlueMatt](https://github.com/TheBlueMatt/llama.cpp/commit/5eb904aaad6cc865bed2b30aad2559ef90558563): the first
  direct GPU-to-GPU Vulkan all-reduce for llama.cpp, on Intel Xe.
- [noon-at-cgn/geforce-bar1-resize](https://github.com/noon-at-cgn/geforce-bar1-resize): the larger memory window of the
  Turing cards.
