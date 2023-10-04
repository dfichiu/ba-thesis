# Chunking with Hyperdimensional Computing, Dynamic Sparse Distributed Memory, and Transformer Attention 

_Learning by chunking_ refers to the process of breaking a large piece of information into small informational units and assembling the bits and pieces of information together to create meaningful concepts, called _chunks_.

Inspired by learning by chunking, we explore the novel possibility of designing a neurally motivated chunking system applied to textual data. We noted that chunks constitute concepts. Therefore, our objective can be equivalently restated as an attempt to construct a system capable of creating concepts from text, which amounts to training, and identifying and extracting the constructed concepts from a given text, which amounts to inference.

To create concepts, we explore two creational avenues: 
1. In the first method, we develop abstract concepts outgoing from [Hofstadter, 2001](http://worrydream.com/refs/Hofstadter%20-%20Analogy%20as%20the%20Core%20of%20Cognition.pdf)’s definition of abstract concepts, who defined concepts as "packets of analogies;"
2. Due to the very experimental nature of the first method, instead of trying to develop concepts, in the second method we construct concepts by directly mining relationships between words captured by the Transformer Attention.[Vaswani et al., 2017](https://arxiv.org/pdf/1706.03762.pdf)

## Project structure
```
.
|-- Makefile
|-- README.md
|-- poetry.lock
|-- pyproject.toml
|-- requirements.txt
`-- src
    |-- deprecated
    |   |-- DSDM.ipynb
    |   |-- DSDM_simplified.ipynb
    |   |-- DSDM_weighted_sup.ipynb
    |   |-- DSDM_weights.ipynb
    |   |-- README.md
    |   |-- Untitled.ipynb
    |   |-- VSA.ipynb
    |   |-- experiment-1.ipynb
    |   |-- hypervector_experiments.ipynb
    |   `-- models
    |       `-- positional.py
    |-- experiments
    |   |-- construct_subsequences.ipynb
    |   |-- dataset_statistics.ipynb
    |   |-- experiment_sliding_window.ipynb
    |   |-- experiment_transformer_sequences.ipynb
    |   |-- results
    |   |   |-- figs
    |   |   |   |-- attention
    |   |   |   `-- dataset-stats
    |   |   `-- html
    |   |-- run_experiment.sh
    |   |-- test_memory.ipynb
    |   |-- test_memory_template.ipynb
    |   |-- train_SWNmemory.py
    |   |-- train_memory.py
    |   |-- visualize_attention.ipynb
    |   `-- visualize_memory.ipynb
    |-- lib
    |   |-- memory
    |   `-- utils
    `-- normalization-experiments
        |-- README.md
        |-- configs
        |-- data
        |-- experiment_template.ipynb
        |-- results
        `-- run_experiment.sh
```




The folder
### Experiments
The [experiments](https://github.com/dfichiu/ba-thesis/tree/master/src/experiments) folder contains the code of the latest experiments. The file types contained are:
* `.ipynb` files: Jupyter notebooks used for short memory training sessions, memory visualization, dataset statistics computation, inference, etc.  For on-the-fly training sessions w/ inference, use
  * [experiment_transformer_sequences.ipynb](https://github.com/dfichiu/ba-thesis/tree/master/src/experiments/experiment_transformer_sequences.ipynb): Train memory by mining Transformer self-attention matrices;
  * [experiment_sliding_window.ipynb](https://github.com/dfichiu/ba-thesis/tree/master/src/experiments/experiment_sliding_window.ipynb): Train memory using the sliding window n-gram method.
* `.py` files: Training scripts for both methods:
  * [train_memory.py](https://github.com/dfichiu/ba-thesis/tree/master/src/experiments/train_memory.py): Train memory by mining Transformer self-attention matrices;
  * [train_SWNmemory.py](https://github.com/dfichiu/ba-thesis/tree/master/src/experiments/train_SWNmemory.py): Train memory using the sliding window n-gram method.
  * Depending on the method, the trained memory and the associated codebook are saved in the subfolders `/memories/method1` (sliding window) or `/memories/method2` (Transformer self-attention) and `/cleanups/method1` (sliding window) or `/cleanup/method2`, respectively.
* `.sh` file [run_experiment.sh](https://github.com/dfichiu/ba-thesis/tree/master/src/experiments/run_experiment.sh): One-click demo of the entire project. The script runs the Jupyter notebooks and saves the created plots and the run notebooks as .html pages. The results produced by the script are saved in the subfolder [results](https://github.com/dfichiu/ba-thesis/tree/master/src/experiments/results). 

### Deprecated
The [deprecated](https://github.com/dfichiu/ba-thesis/tree/master/src/deprecated) folder contains Jupyter notebooks that helped in developing the code for [DSDM](https://github.com/dfichiu/ba-thesis/blob/master/src/lib/memory/DSDM.py) and [Cleanup](https://github.com/dfichiu/ba-thesis/blob/master/src/lib/utils/cleanup.py), the codebook saving the token-hypervector associations. Due to library changes, the notebooks do not run without errors anymore.

### Normalization-experiments
The [normalization-experiments](https://github.com/dfichiu/ba-thesis/tree/master/src/normalization-experiments) folder contains Jupyter notebooks that performed experiments in the context of the sliding window n-gram method with unnormalized and normalized similarity computation. (The script [/src/normaliztion-experiments/run_experiment.sh](https://github.com/dfichiu/ba-thesis/tree/master/src/normalization-experiments/run_experiment.sh) was run with a configuration file from the [/src/normaliztion-experiments/configs](https://github.com/dfichiu/ba-thesis/tree/master/src/normalization-experiments/configs) subfolder; the data for the experiments is located in the subfolder `/data`. The script parametrized the Jupyter notebook `experiment_template.ipynb` based on the chosen configuration file and the resulting .html of the run notebook was saved in the `/results` subfolder.) Due to library changes, the current output has to be reformatted. However, the results of the performed experiments can be seen in [/src/normaliztion-experiments/results](https://github.com/dfichiu/ba-thesis/tree/master/src/normalization-experiments/results). 

## Requirements
The requirements can be found in `requirements.txt`. If you are running the project on the PVS server, activate the conda environment `daniela-py39-clone`.

## One-click demo
To run the one-click demo, once you've installed the requirements/activated the PVS conda environment (the project is located in `dfichiu/ba-thesis`), run [/src/experiments/run_experiment.sh](https://github.com/dfichiu/ba-thesis/tree/master/src/experiments/run_experiment.sh).
The script runs the Jupyter notebooks in [src/experiments](https://github.com/dfichiu/ba-thesis/tree/master/src/experiments) and saves the created plots and the run notebooks as .html pages. The results produced by the script are saved in the subfolder [/src/experiments/results](https://github.com/dfichiu/ba-thesis/tree/master/src/experiments/results). 
