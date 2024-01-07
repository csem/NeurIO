# Canaan Platforms

## List of supported platforms
Few platforms from Canaan are supported, including:

| Platform       | Processor     | Description               | Status    | Firmware     | Tools version |
|----------------|---------------|---------------------------|-----------|--------------|---------------|
| Sipeed Maix M1 | Kendryte K210 | 32-bit RISC-V CPU and KPU | Supported | MaixPy 0.6.2 | NCC 0.2.0     |

## Kendryte K210

### Installation

#### Prerequisites

To install NeurIO for Kendryte K210, you need to install the following tools:
- NCC (NNCase from Canaan): a compiler for neural networks, transforming TFLite models to K210 compatible models. The link to the version 0.2.0 is ["here"](https://github.com/kendryte/nncase/releases/download/v0.2.0-beta4/ncc_osx_x86_64.dmg).
- MaixPy: a Python interpreter for Kendryte K210. The link to the version 0.6.2 is ["here"](https://dl.sipeed.com/shareURL/MAIX/MaixPy/release/master).
- Instructions to install MaixPy on the Sipeed Maix M1 board are ["here"](https://maixpy.sipeed.com/en/get_started/).

#### Installation steps
1. Download and install NNCase in neurio. Extract the zip file and copy the `ncc` executable to the `neurio/converters/ncc` folder.
2. Download the firmware from Sipeed.
3. Flash the MaixPy firmware on the Sipeed Maix M1 board.
4. Connect the Sipeed Maix M1 board to the computer.

### Usage

To use NeurIO and benchmark models on Kendryte K210, you need to connect the board to the computer and run the following command:
```python
from neurio.devices.canaan import K210

model = ...
input_data = ...
port = "/dev/tty.someport" # port to which the device is connected
device = K210(port=port, log_dir=None)

device.prepare_for_inference(model=model)
predictions = device.predict(input_x=input_data, batch_size=2)
```

#### Execution process

The execution process for K210 is as follows:

```{mermaid}
%%{init: {'theme':'dark'}}%%
    flowchart TD
        subgraph PR["PREDICTION"]
            subgraph SP["Setup"]
                A["Download NCC"]
                B["Convert model to TFLite"]
                C["Convert TFLite model to K210"]
                D["Generate scripts for inference"]
                E["Upload scripts"]
                F["Upload model"]
                A --> B
                B --> C
                C --> D
                D --> E
                E --> F
            end
            
            subgraph INF["Inference"]
                G["Save data"]
                H["Upload batch of data"]
                I["Run inference"]
                J["Save results to SD"]
                K["Download results"]
                G --> H
                H --> I
                I --> J
                J --> K
                K -- if more data --> H
            end
            
            subgraph PP["Post-processing"]
                L["Aggregate results"]
                M["Compute metrics"]
                N["Finished"]
                L --> M --> N
            end
            
            SP --> INF
            INF -- if all data predicted --> PP
        
        end
        
 ```
