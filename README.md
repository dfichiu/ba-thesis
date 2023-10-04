# Chunking with Hyperdimensional Computing, Dynamic Sparse Distributed Memory, and Transformer Attention 

_Learning by chunking_ refers to the process of breaking a large piece of information into small informational units and assembling the bits and pieces of information together to create meaningful concepts, called _chunks_.

Inspired by learning by chunking, we explore the novel possibility of designing a neurally motivated chunking system applied to textual data. We noted that chunks constitute concepts. Therefore, our objective can be equivalently restated as an attempt to construct a system capable of creating concepts from text, which amounts to training, and identifying and extracting the constructed concepts from a given text, which amounts to inference.

## Project structure
```

```




The folder
### Experiments

### Deprecated
Contains Jupyter notebooks that helped in developing the code for [DSDM]() and [Cleanup](), the codebook saving the token-hypervector associations. Due to library changes, the notebooks do not run without errors anymore.

### Normalization-experiments
Contains Jupyter notebooks that performed experiments in the contxt of the sliding window n-gram method with unnormalized and normalized similarities. (The script `run_experiment.sh` was run with a configuration file from the `/configs` subfolder. The sscript parametrized the Jupyter notebook `experiment_template.ipyng` based on the chosen configuration file and the resulting .html of the run notebook was saved in the `/results` subfolder.) Due to library changes, the output has to be reformatted. However, the results of the experiments can be seen in the `results` subfolder

## Requirements
The requirements can be found in `requirements.txt`.

