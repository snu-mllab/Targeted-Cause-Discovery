from slurm_launcher.sbatch_launcher import launch_tasks


def run():
    gpu_option = 2

    for data_level in [
        ("dropout",),  # for continuous model
            # ("dropout", "dropout_poisson500", "dropout_poisson1"),  # for discrete model (use n_env 20)
    ]:
        for graphs in [
            ("er", "sf", "sbm", "sf_indirect", "sf_in"),
        ]:
            base_cmd = "python -B src/train.py --cause --no_tqdm --not_save_last "

            base_cmd += f"--data_file "
            for g in graphs:
                for e in [2, 4, 6]:
                    base_cmd += f"train_sergio_syn/{g}_e{e}_add "

            base_cmd += "--data_level "
            for d in data_level:
                base_cmd += f"{d} "

            tag = f"gene_dropout"

            base_cmd += f"--run_name slurm_{tag} "
            param_dict1 = {
                "--epoch": [200],
                # data
                "--n_env": [50],
                "--num_vars": [200],
                "--obs_size": [200],
                "--obs_ratio": [0.05],
                "--dtype": ["float32"],
                # normalization
                "--shift": [12],
                # model
                "-l": [10],
                "--embed_dim": [16],
                "--n_head": [16],
                "--ffn_embed_dim": [96],
                # optimizer
                "--lr": [6e-4, 8e-4, 1e-3],
                "--batch_size": [16],
                "--accumulate_batches": [2],
            }

            param_dict_list = [param_dict1]
            job_name_list = [tag]

            for job_name, param_dict in zip(job_name_list, param_dict_list):
                launch_tasks(param_option=gpu_option,
                             base_cmd=base_cmd,
                             param_dict=param_dict,
                             partition='rtx3090',
                             exclude='nutella,alan,ohm,carl,euclid,hamming',
                             timeout="1-00:00:00",
                             job_name=job_name,
                             max_job_num=200)


if __name__ == '__main__':
    run()
