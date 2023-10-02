This file contains the initial experiments performed in the context of the sliding window n-gram method to understand how normalization in the computation of the similarity affects the results. The experiments required a .yaml file for configuration and were run by running the `run_experiment.sh` script with the option `-y` with the argument the (relative) path to the configuration .yaml to be used, e.g., 

```python
./run_experiment.sh -y configs/experiment-6
```

The `run_experiment.sh` uses the [paprmill](https://papermill.readthedocs.io/en/latest/) library parametrize the jupyter notebook `experiment_templat.ipynb`, i.e., to create a new notebook where the parameters read from the configuration .yaml are read in a jupyter cell tagged with the `parameters` tag. The newly created notebook is then and the run notebook is saved as an .html file. Finally, the creted notebook is deleted.


Since the experimnts were performed before the [library](https://github.com/dfichiu/ba-thesis/tree/master/src/lib) was refactored, running the experiments now would require some additional processing of the results. However, the results of the first five experiments performed before the library refactoring are saved as .html files in the `results` subfolder.