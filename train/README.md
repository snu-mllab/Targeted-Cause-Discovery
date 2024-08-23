## Trining codes for TCD-DL

This is the repository for training codes of TCD-DL. Due to storage limitations, we are currently not distributing the training data for our model. Please use the training code for reference only.

To obtain the simulation data, please refer to [`./simulator`](https://github.com/snu-mllab/Neural-Relation-Graph/blob/main/train/simulator).

We train our model on a single RTX3090 by running 
```
python slurm_train.py
```
which is identical to
```
python -B src/train.py --cause --no_tqdm --not_save_last \
--data_file [list of data folders] \
--data_level dropout \
--n_env 50 --epoch 200 \
--batch_size 16 --accumulate_batches 2 --lr 8e-4
```
- Set `DATAPATH` in `src/args.py` to point the directory containing the [list of data folders] above (reference: args.py L136). 
- Each data folder has the following structure 
    ```
    ecoli
    ㄴ 0
    ㄴ 0
    ...
    ㄴ 49
      ㄴ DAG.npy (causal graph)
      ㄴ [data_level].npy (observation)
      ㄴ [data_level]_intv.npy (intervention matrix)
    ```
- Data_level represents the simulation fidelity. Please check [`./simulator`](https://github.com/snu-mllab/Neural-Relation-Graph/blob/main/train/simulator).
- For discrete and impute models, we utilize the mixture of data sources across multiple [data_level]. Please check `./slurm_train.py` L9.