# ST Platforms

## List of supported platforms
Few platforms from Canaan are supported, including:

| Platform       | Processor     | Description                      | Status    | Firmware | Tools version                                                             |
|----------------|---------------|----------------------------------|-----------|----------|---------------------------------------------------------------------------|
| NUCLEO-144     | ARM Cortex M7 | Nucleo Dev Board (NUCLEO-H723ZG) | Supported | 1.10     | X-Cube-AI 7.1.0, STMCubeIDE 6.5.0, STMProgrammerCLI 2.10.0, ST-Link 2.0.2 |
| STML4R9I-DISCO | ARM Cortex M4 | Discovery Board (STM32L4R9/S9)   | Supported | 1.10     | X-Cube-AI 7.1.0, STMCubeIDE 6.5.0, STMProgrammerCLI 2.10.0, ST-Link 2.0.2 |

## STM32 M4 and M7 Family

### Installation

#### Prerequisites

To install NeurIO for M4 and M7, you need to install the following tools:
- X-Cube-AI: a tool to convert TFLite models to STM32 compatible models. The link to the version 7.1.0 is [here](https://www.st.com/en/embedded-software/x-cube-ai.html).
- STMCubeIDE: an IDE to develop and flash firmware on STM32 boards. The link to the version 6.5.0 is [here](https://www.st.com/en/development-tools/stm32cubeide.html).
- STMProgrammerCLI: a tool to flash firmware on STM32 boards. The link to the version 2.0.1 is [here](https://www.st.com/en/development-tools/stm32cubeprog.html).
- ST-LINK Utility: a tool to flash firmware on STM32 boards. The link to the version 2.0.2 is [here](https://www.st.com/en/development-tools/stsw-link004.html).

#### Installation steps

1. Download and install the prerequisite tools.
2. Setup the environment variables to set the path to the STM32 tools. 
Example for MacOS Sonoma:

```bash
export STM32CUBEPROGRAMER_CLI_PATH=/Applications/STMicroelectronics/STM32Cube/STM32CubeProgrammer/STM32CubeProgrammer.app/Contents/MacOs/bin/STM32_Programmer_CLI`
export X_CUBE_AI_PATH=/Users/alex/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/7.1.0
export STM32CUBEIDE_PATH=/Applications/STM32CubeIDE.app/Contents/MacOS/STM32CubeIDE
```

### Usage

You can import a device from the ST module and use it to benchmark models on STM32 boards.
For example, to run a model on the NUCLEO-H723ZG board, you can run the following code:

```python
from neurio.devices.st import NUCLEOH723ZG

model = ...
input_data = ...
port = "serial" # port to which the device is connected
device = NUCLEOH723ZG(port=port, log_dir=None)

device.prepare_for_inference(model=model)
predictions = device.predict(input_x=input_data, batch_size=2)
```

#### Execution process

The execution process for STM32 M4 and M7 families is as follows:

```{mermaid}
%%{init: {'theme':'dark'}}%%
    flowchart TD
        subgraph PR["PREDICTION"]
            
            subgraph SP["Setup"]
                              
                B["Convert model to TFLite<br/>(X-CUBE-AI)"]
                C["Convert TFLite model to C-Code<br/>(X-CUBE-AI)"]
                D["Generate scripts for inference<br/>(X-CUBE-AI)"]
                E["Compile files to .elf file<br/>(STMCubeIDE)"]
                F["Upload elf using ST-Link<br/>(STMCubeProgrammer)"]
                B --> D
                D --> C
                C --> E
                E --> F
            end
            
            subgraph INF["Inference"]
                
                G["Save data locally"]
                H["Upload batch of data to device memory<br/>(STMCubeProgrammer)"]
                I["Run inference<br/>(X-CUBE-AI)"]
                J["Download results from device memory<br/>(STMCubeProgrammer)"]
                K["Retrieve Profiler values from AIRunner<br/>(X-CUBE-AI)"]
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