# ST Platforms

## List of supported platforms
Few platforms from Canaan are supported, including:

| Platform       | Processor     | Description          | Status    | Firmware | Tools version                  |
|----------------|---------------|----------------------|-----------|----------|--------------------------------|
| Coral Edge TPU | Google Coral  | Google Coral TPU USB | Supported | -        | TFLite Runtime, PyCoral v2.0.0 |

## Google Coral Edge TPU

### Installation

#### Prerequisites

To use Google Edge TPU with NeurIO, you need to follow the installation steps on the [official documentation](https://coral.ai/docs/accelerator/get-started/#requirements).

#### Installation steps

The current PyCoral library (v2.0.0) conflicts with the official Tensorflow package. To use the Edge TPU,
it is required to modify the official tflite interpreter scripts. To do so, you need to modify the following file

```bash
{your_venv}/lib/python3.x/site-packages/tflite_runtime/interpreter.py
```
 by replacing the following line:
```python
from tensorflow.lite.python.interpreter_wrapper import _pywrap_tensorflow_interpreter_wrapper as _interpreter_wrapper
```
with:
```python
try:
    from tensorflow.lite.python.interpreter_wrapper import _pywrap_tensorflow_interpreter_wrapper as _interpreter_wrapper
except:
    print("Using Interpreter from tflite_runtime package.")
```

As a result, the interpreter of Tensorflow Lite will be bypassed by the interpreter of the TFLite Runtime of PyCoral.


### Usage

You can import the Google Coral EdgeTPU device from the `devices.google.coral` module and use it to benchmark models.
For example, you can run the following code:

```python
from neurio.devices.google.coral import EdgeTPU

model = ...
input_data = ...
device = EdgeTPU(port=None, log_dir=None)

device.prepare_for_inference(model=model)
predictions = device.predict(input_x=input_data, batch_size=1)
```

#### Execution process

The execution process for EdgeTPU is as follows:

```{mermaid}
%%{init: {'theme':'dark'}}%%
    flowchart TD
        subgraph PR["PREDICTION"]
            
            subgraph SP["Setup"]
                              
                B["Convert model to TFLite<br/>(Tensorflow Lite)"]
                C["Save TFlite model to file"]
                D["Allocate tensors of model on EdgeTPU<br/>(PyCoral)"]
                B --> C --> D
            end
            
            subgraph INF["Inference"]
                F["Preprocess data<br/>(convert to uint8 if necessary)"]
                G["Save data locally"]
                H["Fill input tensors<br/>memory with data"]
                I["Run inference"]
                J["Read output tensor memory"]
                K["Fill Profiler with metrics from inference"]
                F --> G --> H
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