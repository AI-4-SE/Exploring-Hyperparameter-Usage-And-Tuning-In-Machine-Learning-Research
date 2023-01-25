# Exploring Hyperparameter Usage and Tuning in Machine Learning Research

This repository provides additional material for our paper: "Explroing Hyperparameter Usage and Tuning in Machine Learning Research".


## API Crawler

We developed an API crawler for three popular and widely used ML libraries, namely scikit-learn, TensorFlow and Pytorch. The scripts can be found [here](https://anonymous.4open.science/r/ml-config-options-75C1/).

## Code Repository Analysis

Note that we developed plugins for the each ML library, which apply static code-analysis and control- and data-flow analyis to locate API calls from the corresponding library and extract their configuration settings. The plugins are integrated into the CfgNet. Its implementation can be found [here](https://anonymous.4open.science/r/CfgNet-3D67/). Our analysis script relies on the CfgNet and assumes that it is run on our Slurm cluster if the hostname is `tesla` or starts with `brown`. You can find our evaluation script in [`analysis/`](analysis).

You can start the analysis by running `run.sh`.
It takes an optional parameter which is a Git tree-ish (e.g. `main`) that can be used to get a certain version of CfgNet.

**For this analysis, it is required to use the `ml` branch of the CfgNet and to pass the `--enable-ml-plugins (-m)` argument, because only this branch contains the ML library plugins and extractes the API calls.**

The result files will be in `results/`.
You can find the modified repositories in `out/`.


## Data and Scripts

The [data/](data/) directory contains all the data used in this paper, while the [src/](src/) directory contains all scripts used to process the data.

- [data/dblp/](data/dblp/) : contains the data crawled from the DBLP digital bibliography
- [data/library_data/](data/library_data/): contain the API data of the ML libraries
- [data/paper_analysis/](data/paper_analysis/): contains the metadata for each paper and the data used for measuring the annotator agreement
- [data/statistics/](data/statistics/): contains the data extracted from the code repositories

- [src/cross_validation/](src/cross-validation/): contains the script to calculate the inter-annotator agreement
- [src/dblp_results/](src/dblp_results/): contains the script the calculate the number of papers dealing with hyperparameter importance and tuning
- [src/library_stats/](src/library_stats//): contains the script the calculate the total number of API calls and parameter for each library
- [src/](src): contains the scripts to process the data extracted from the code repostories and respective research papers