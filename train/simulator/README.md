## Simulator

This is the repository for SERGIO simulator. The codes are borrowed from the [AVICI repository](https://github.com/larslorch/avici/tree/full).

To run the codes, please install the requirements.txt following the reference AVICI repository.

To obtain the simulation data, run
```
python -B -m avici.experiment.data --data_config_path ./experiments/gene-cause/[config file] --path_data [output path] -j [env index]
```
- Please refer to the data configurations in [`./experiments/gene-cause`](https://github.com/snu-mllab/Targeted-Cause-Discovery/tree/main/train/simulator/experiments/gene-cause).
- We train our models on {er,sbm,sf,sf_in,sf_indirect}_e{2,4,6}. 
- The test configs are {ecoli,yeast} including OOD settings {func,func_noise}.
- We generates 50 environments (`-j 0~49`) for each config (reference: [`slurm.py`](https://github.com/snu-mllab/Targeted-Cause-Discovery/blob/main/train/simulator/slurm.py)).

The resulting files contain
1. ground-truth causal graph `DAG.npy`.  
2. observation data:  
(1) clean data without any technical noise `clean.npy`.  
(2) continuous expression data `dropout.npy` (i.e., high fidelity).  
(3) discretized UMI count expression data `data.npy`  (i.e., low fidelity).
5. corresponding intervention matrices `*_intv.npy`.

To obtain the medium fidelity data, please check `slurm_poisson.py`, which executes `poisson.py`. You need to change the `base_path` in `poisson.py` to [output path] (`--path_data`) in the command above.
