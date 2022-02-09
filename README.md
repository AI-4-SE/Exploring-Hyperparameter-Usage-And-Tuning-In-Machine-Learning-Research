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
- Are default values used for ML projects? For which ML technique?
- When do default values diverge? Any specific scenario/task?
- What value range is used for specific ML tecniques?