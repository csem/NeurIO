# Generic Deployment

This section describes the generic deployment process for a deep learning model. The process is divided into two main
parts: System Preparation and Prediction. The first part describes the steps required to prepare the system for the
deployment of the model. The second part describes the steps required to perform the prediction.

## System Preparation

The system preparation part is divided into four main steps: Model Training, Model Preparation, Inference Code
Generation and Deployment. The first step is the data collection and model training. This step is performed on a
workstation and is not part of NeurIO.

The second step is the model preparation. This step is performed on a per-device basis, as it requires the optimization
and quantization to be configured for the specific device.

The third step is the inference code generation. It also depends on the device, as it requires the code to be generated
using the tools and SDK libraries provided by the device manufacturer.

The fourth step is the deployment, which is the transfer of the model and the executable code to the device. This step
is also device-specific, as it depends on the device’s communication protocol and the tools provided by the device
manufacturer.

```{mermaid}
%%{init: {'theme':'dark'}}%%
flowchart TD
    subgraph SP["SYSTEM PREPARATION"]
        subgraph MT["Model training"]
            A0["Data collection"]
            B0["Data annotation"]
            C0["Data preprocessing"]
            D0["Model Definition"]
            A["Model training"]
            B["Model evaluation"]
            A0 --> B0 --> C0 --> D0 --> A --> B
        end

        subgraph MP["Model preparation"]
            C["Model optimization"]
            D["Model quantisation"]
            E["Model conversion"]
            C --> D --> E
        end

        subgraph IC["Inference code generation"]
            G["Inference script generation"]
            H["[optional] Compilation"]
            G --> H
        end

        subgraph DP["Deployment "]
            I["Transfer of model to device"]
            K["Transfer of executable code to device"]
            I --> K
        end
        
        MT --> MP --> IC --> DP
    end
```

## Prediction

The prediction part is divided into three main steps: Data Preparation, Inference and Reporting. The first step is the
data preparation, which is the process of preparing the data for the inference. Usually, the preprocessing occurs on the
workstation, as it comes at an additional cost for the hardware. However, the preprocessing is highly dependent on the
optimization of the model, and therefore is performed by NeurIO inside the device class. The data is then transferred to
the device memory using the device-specific communication protocol.

The second step is the inference, which is the process of running the model on the device. This step is performed on the
device, as it requires the model to be loaded into the device memory. The inference is performed on the batch of data
previously uploaded, as the storage of the device is limited. During the inference, the metadata (energy, cycles, etc…)
is collected and stored in the device memory. Both the inference results and the metadata are then transferred to the
workstation.

Finally, then inference results are aggregated by the workstation, and the KPIs are computed and provided back to the
user. The KPIs can then be displayed on the workstation or stored in a database.

```{mermaid}
%%{init: {'theme':'dark'}}%%
    flowchart TD
        subgraph PR["PREDICTION"]
            subgraph DT["Data preparation"]
                L["Data Preprocessing"]
                N["Subsample batch"]
                O["Data transfer to device memory"]
                L --> N --> O
            end
    
            subgraph IR["Inference"]
                P["Load model"]
                Q["Load batch data"]
                RI["Run inference"]
                ME["Measure inference"]
                ST["Save predictions"]
                S["Transfer results to workstation"]
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
