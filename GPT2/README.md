
##  How to Run

###  Language & Framework

* **Language**: Python
* **Framework**: PyTorch

---

###  Installation

Make sure you have Python installed, then run the following commands to set up the environment:

```bash
## Installation

To install the package without ML dependencies:
pip install sagar-neuron-gpt2

Then you should install
pip install torch
pip install tiktoken
pip install numpy


To include ML dependencies (torch and tiktoken):
pip install sagar-neuron-gpt2[ml]
pip install numpy



```

---

###  Train the Model

To train and save your GPT-2 model weights, run the following:

```python
from sagar_nueron_gpt2.TrainAndSaveGptWeights import Execute

exe = Execute()
exe.execute()
```

---

###  Inference from Trained Model

To run inference using the model you trained:

```python
from sagar_nueron_gpt2.inference_model import Inferencing

exe = Inferencing()
exe.inference()
```

---


