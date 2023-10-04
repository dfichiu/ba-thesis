# Chunking with Hyperdimensional Computing, Dynamic Sparse Distributed Memory, and Transformer Attention 

_Learning by chunking_ refers to the process of breaking a large piece of information into small informational units and assembling the bits and pieces of information together to create meaningful concepts, called _chunks_.

Inspired by learning by chunking, we explore the novel possibility of designing a neurally motivated chunking system applied to textual data. We noted that chunks constitute concepts. Therefore, our objective can be equivalently restated as an attempt to construct a system capable of creating concepts from text, which amounts to training, and identifying and extracting the constructed concepts from a given text, which amounts to inference.

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
The `experiments` folder contains the code of the latest experiments. The file types contained are
* `.ipynb` files: Jupyter notebooks used for short memory training sessions, memory visualization, dataset statistics computation, inference, etc.  For on-the-fly training sessions w/ inference, use
  * `experiment_transformer_sequences.ipynb`: Train memory by mining Transformer self-attention matrices;
  * `experiment_sliding_window.ipynb`: Train memory using the sliding window n-gram method.
* `.py` files: Training scripts for both methods:
  * `train_memory.py`: Train memory by mining Transformer self-attention matrices;
  * `train_SWNmemory.ipynb`: Train memory using the sliding window n-gram method.
  * Depending on the method, the trained memory and the associated codebook are saved in the subfolders `/memories/method1` (sliding window) or `/memories/method2` (Transformer self-attention) and `/cleanups/method1` (sliding window) or `/cleanup/method2`, respectively.
* `.sh` file: One-click-demo of the entire project. The script runs the Jupyter notebooks, saves the created plots and the run notebook as .html pages. The results produced by the script are saved in the subfolder `/results`. 

### Deprecated
The `deprecated` folder contains Jupyter notebooks that helped in developing the code for [DSDM](https://github.com/dfichiu/ba-thesis/blob/master/src/lib/memory/DSDM.py) and [Cleanup](https://github.com/dfichiu/ba-thesis/blob/master/src/lib/utils/cleanup.py), the codebook saving the token-hypervector associations. Due to library changes, the notebooks do not run without errors anymore.

### Normalization-experiments
The `normalization-experimentss` folder contains Jupyter notebooks that performed experiments in the context of the sliding window n-gram method with unnormalized and normalized similarity computation. (The script `run_experiment.sh` was run with a configuration file from the `/configs` subfolder; the data for the experiments is located in the subfolder `/data`. The script parametrized the Jupyter notebook `experiment_template.ipynb` based on the chosen configuration file and the resulting .html of the run notebook was saved in the `/results` subfolder.) Due to library changes, the current output has to be reformatted. However, the results of the performed experiments can be seen in the `results` subfolder

## Requirements
The requirements can be found in `requirements.txt`.

