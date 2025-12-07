
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
from sagar_neuron_gpt2.TrainAndSaveGptWeights import Execute

exe = Execute()
exe.execute()
```

---

###  Inference from Trained Model

To run inference using the model you trained:

```python
from sagar_neuron_gpt2.inference_model import Inferencing

exe = Inferencing()
exe.inference()
```

---

---


###  Binary Classifiers via Transfomers and Fourier Transformers[Not published to pypi hub]

To Run Binary classifer demo, run the following:

```python

from  ClassifierGPT import ClassifierGPT

if __name__ == "__main__":
    gpt2 = ClassifierGPT()
    gpt2.training_script_normal_gpt()
    gpt2.training_script_fourrier()


```

---


