# How it works

MiniMax-H3-Image produces a 1024×1024 image with a 50-layer diffusion transformer, six denoising steps and a 24-layer ViT decoder. A latent has 24 channels on a 64×64 grid. Each 2×2 latent patch occupies one image token, for 1,024 image tokens. Up to 256 text tokens share the sequence.

## Text encoder

The CPU reads the Qwen3-VL token embedding rows referenced by the prompt and streams layers 0–49 from the safetensors shards. Tokenization uses the image model's tokenizer without special tokens or a chat template. Windows unbuffered reads load the next layer's bf16 tensors while the current layer runs in fp32. The refiner loads its condition projection and two transformer layers from `refiner/weights.safetensors` and prepares the 5,120-channel encoder output for the DiT. It also prepares the position embeddings and modulation tags used in denoising.

## Transformer

Each layer has four int4 projections and one DynamicDispatch attention session. The host runs the RMS normalization outside attention, modulation, SwiGLU activation and residual additions.

| Projection | Full input → output width | NPU split |
|---|---|---|
| qkv | 5,376 → 21,504 | Input width |
| out | 7,168 → 5,376 | Output width |
| fc1 | 5,376 → 28,672 | Input width |
| fc2 | 14,336 → 5,376 | Output width |

Asymmetric int4 quantization uses 128 input channels per group and selects the lowest squared error among nine clip ranges. The packer pads channels to shapes supported by the Ryzen AI Software kernels. The host casts each 256-row input tile to bf16 and converts the output pieces to fp32. It sums the input-width partial results in fp32 in their original order. Each projection processes five tiles for the fixed 1,280-row frame.

Attention uses one fixed-shape SD DynamicDispatch graph three times, for groups of 24, 24 and 8 heads. The host interleaves the rotary channel pairs and pads the last group to 24 heads. The graph applies Q/K RMS normalization and RoPE. It computes attention on both V and a validity mark. Dividing the first result by the second removes the contribution of padded rows to the softmax denominator.

CPU tensors provide the timestep embeddings and the six modulation arrays. At each step the CPU projects the latent patches, places the text features in front of the image sequence, applies the step's output head, and advances the latent with the velocity-form schedule.

### Session lifetime and processor affinity

The runtime creates, runs and releases one layer's projections and attention at a time. It keeps one SD DynamicDispatch session alive in each NPU process. In the layer residency experiment, one additional resident transformer layer used about 1.26 GB of NPU shared memory. Shared memory was 1.76 GB with zero resident layers and 9.32 GB with six. The machine exposed 15.9 GB of NPU shared memory. One-layer streaming fits this budget.

The first projection call after session creation can take longer as more sessions are created in a process. The transformer starts after the text worker exits and releases its weights. By default a fresh worker takes over after step 3, starting from a saved latent. Imports can overlap during denoising. Session creation waits for the previous worker to exit. The decoder follows the same rule.

On a single processor group with unequal EfficiencyClass values, the generation process selects the highest class and its children inherit the affinity. The Ryzen AI 7 PRO 350 used for verification has eight Zen 5 and eight Zen 5c logical processors. A homogeneous CPU keeps its inherited affinity. The generator requests display and system execution state for the duration of the run and checks available physical memory.

## Decoder

A 64×64 latent is denormalized and decoded as a 5×5 grid of 16×16 latent tiles. Each produces a 256×256 image tile. The decoder batches the 25 tiles through one layer at a time, runs the four int4 linear projections on the NPU and computes attention on the CPU. Neighboring image tiles are blended across 64 pixels to produce the 1024×1024 RGB image. The CPU option uses the Diffusers fp32 decoder on the same tile grid.

## Conversion and runtime limits

Ryzen AI Software supplies instruction streams for specific tensor dimensions. A single projection at the transformer's original widths does not fit the available kernel shapes. CPU-side concatenation and addition assemble the supported subprojections. The 56-head attention likewise uses the available 24-head SD kernel.

Concurrent NPU work in separate generation processes has produced invalid outputs in testing. Sustaining all layer sessions would exceed the shared memory available on the verified machine. SD DynamicDispatch sessions can also fault during normal native-library shutdown. An NPU worker saves its result and exit status before ending through the Windows process API.

Setup downloads the compiled attention graphs and constants, packed projections, and CPU tensors from a fixed Hugging Face commit, then checks each file against the SHA256 manifest. For local builds, the converter creates each unit in its working directory, checks its required files, then moves the finished unit into the build directory. On restart, it clears temporary units and checks completed file sizes before reusing them. The native partitioner is called through an ASCII directory junction when the output path contains non-ASCII characters.
