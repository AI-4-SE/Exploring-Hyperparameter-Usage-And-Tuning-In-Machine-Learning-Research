# ml-settings

## Evaluation

The evaluation script assumes that it's run on our Slurm cluster if the hostname is `tesla` or starts with `brown`.

You can find our evaluation script in [`eval/`](eval).

You can start the evaluation by running `run.sh`.
It takes an optional parameter which is a Git tree-ish (e.g. `main`) that can be used to get a certain version of CfgNet.

**For this evaluation, it is required to use the `sklearn` branch of the CfgNet, because only this branch contains the sklearn plugin and computes the statistics.**

The result files will be in `results/`.
You can find the modified repositories in `out/`.


## Research Question
- How is Machine Learning Code of common ML libraries configured?
    - Which values are assigned to options that initialize ML algorithms?
    - What kind of values are used? Default values or values optimized with hyperparameter tuning?
    - When do these values diverge? Are specific scenarios/tasks responsible?
    - What is the value range fpr specific options?