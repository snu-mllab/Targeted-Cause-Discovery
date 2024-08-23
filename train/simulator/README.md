## Simulator

This is the repository for SERGIO simulator. The codes are borrowed from the [AVICI repository](https://github.com/larslorch/avici/tree/full).

To run the codes, please install the requirements.txt following the reference AVICI repository.

To obtain the simulation data, run
```
python -B -m avici.experiment.data --data_config_path ./experiments/gene-cause/[config file] --path_data [output path] -j [env index]
```
- Please refer to the data configurations in `./experiments/gene-cause`
- We train our models on {er,sbm,sf,sf_in,sf_indirect}_e{2,4,6}
- The test configs are {ecoli,yeast} including OOD settings {func,func_noise}.
- We generates 50 environments (`-j 0~49`) for each config.

The resulting files contain
1. ground-truth causal graph `DAG.npy`.
2. clean data without any technical noise `clean.npy`.
3. continuous expression data `dropout.npy` (i.e., high fidelity).
4. discretized UMI count expression data `data.npy`  (i.e., low fidelity).
5. corresponding intervention matrices `*_intv.npy`.

To obtain the medium fidelity data, please follow `slurm_poisson.py` which runs `poisson.py`.