# Canaan Platforms

## List of supported platforms
Few platforms from Canaan are supported, including:

| Platform       | Processor     | Description               | Status    | Firmware     | Tools version |
|----------------|---------------|---------------------------|-----------|--------------|---------------|
| Sipeed Maix M1 | Kendryte K210 | 32-bit RISC-V CPU and KPU | Supported | MaixPy 0.6.2 | NCC 0.2.0     |

## Kendryte K210

### Installation

#### Prerequisites

To install NeurIO on Kendryte K210, you need to install the following tools:
- NCC (NNCase from Canaan): a compiler for neural networks, transforming TFLite models to K210 compatible models. The link to the version 0.2.0 is [here](https://github.com/kendryte/nncase/releases/download/v0.2.0-beta4/ncc_osx_x86_64.dmg).
- MaixPy: a Python interpreter for Kendryte K210. The link to the version 0.6.2 is [here](https://dl.sipeed.com/shareURL/MAIX/MaixPy/release/master).
- Instructions to install MaixPy on the Sipeed Maix M1 board are [here](https://maixpy.sipeed.com/en/get_started/).

#### Installation steps
1. Download and install NNCase in neurio. Extract the zip file and copy the `ncc` executable to the `neurio/converters/ncc` folder.
2. Download the firmware from Sipeed.
3. Flash the MaixPy firmware on the Sipeed Maix M1 board.
4. Connect the Sipeed Maix M1 board to the computer.

### Usage

To use NeurIO and benchmark models on Kendryte K210, you need to connect the board to the computer and run the following command:
```bash

```

#### Execution process

The execution process for K210 is the following

<!DOCTYPE html>
<html lang="en">
  <body>
    <pre class="mermaid">
    flowchart TB
    A[Download NCC]
        B[Convert model to TFLite]
        C[Convert TFLite model to K210]
        D[Generate scripts for inference]
        E[Upload scripts]
        F[Upload model]
        G[Save data]
        H[Upload batch of data]
        I[Run inference]
        J[Save results to SD]
        K[Download results]
        L[Aggregate results]
        M[Compute metrics]
        N[Finished]
        subgraph setup
            direction TB
            A --> B
            B --> C
            C --> D
            D --> E
            E --> F
        end
        subgraph inference
            direction TB
            G --> H
            H --> I
            I --> J
            J --> K
            K -- if more data --> H
        end
        subgraph post-processing
            direction TB
            L --> M
            M --> N
        end;
        F --> G
        K -- if all data predicted --> L
    </pre>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    </script>
  </body>
</html>
