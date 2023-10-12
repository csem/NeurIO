# Template for device documentation

This template can be used to create the documentation for a new device.

## Introduction

This section describes the device and its main characteristics.

### Device Characteristics

Describe the technology used by the device, as well as the manufacturer and the device name.
Any information about the device (bus speed, memory) is welcome.


## Installation

This steps describe what is needed to install the device and use it with NeurIO.
This might include downloading external libraries or tools.

### Prerequisites

To use {DEVICE} with NeurIO, you need to install the following tools:
- tool1
- tool2

#### Installation steps
1. step1
2. step2

### Usage

To use NeurIO and benchmark models on {DEVICE}, you need to connect the board to the computer and run the following command:
```bash
TODO
```

### Execution process

The execution process for {DEVICE} is the following

#### System preparation 

```{mermaid}
%%{init: {'theme':'dark'}}%%

flowchart TD
    classDef tool fill:#444,stroke-dasharray:3;

    subgraph SP["SYSTEM PREPARATION"]

        subgraph MP["Model preparation"]
           subgraph C["Model optimization"]
           Tool1:::tool
           Tool2:::tool
           end
           subgraph D["Model quantisation"]
           Tool3:::tool
           end
           subgraph E["Model conversion"]
           Tool4:::tool
           end
            C --> D --> E
        end

        subgraph IC["Inference code generation"]
            subgraph G["Inference script generation"]
            ToolA:::tool
            end
            subgraph H["[optional] Compilation"]
            ToolB:::tool
            end
            G --> H
        end

        subgraph DP["Deployment "]
            subgraph I["Transfer of model to device"]
            ToolX:::tool
            end
            subgraph K["Transfer of executable code to device"]
            ToolY:::tool
            end
            I --> K
        end
        
        MP --> IC --> DP
    end
```

### Inference

```{mermaid}
%%{init: {'theme':'dark'}}%%

flowchart TD
    classDef tool fill:#444,stroke-dasharray:3;

        T1:::tool

        subgraph PR["PREDICTION"]
            subgraph DT["Data preparation"]
                subgraph L["Data Preprocessing"]
                 T1[Tool1]:::tool
                end
                subgraph N["Subsample batch"]
                 T3[Tool1]:::tool
                 T4[Tool2]:::tool
                end
                subgraph O["Data transfer to device memory"]
                 T5[Tool1]:::tool
                end
                L --> N --> O
            end
    
            subgraph IR["Inference"]
                subgraph P["Load model"]
                    T6[Tool1]:::tool
                end
                subgraph Q["Load batch data"]
                    T7[Tool1]:::tool
                end
                subgraph RI["Run inference"]
                    T8[Tool1]:::tool
                end
                subgraph ME["Measure inference"]
                    T9[Tool1]:::tool
                end
                subgraph ST["Save predictions"]
                    T10[Tool2]:::tool
                end
                subgraph S["Transfer results to workstation"]
                    T11[Tool1]:::tool 
                end
                RI --> ME --> S
                P --> Q --> RI --> ST --> S
                
            end
            
            subgraph KPI["Reporting"]
                T["Aggregate results"]
                U["Compute KPI"]
                V["Display KPI"]
                T --> U --> V
            end
            
            DT --> IR --> KPI
        end
```