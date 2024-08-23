from slurm_launcher.sbatch_launcher import launch_tasks


def run():
    base_cmd = "python -B run.py --save_pred "

    param_dict = {
        "--data_file": ["train_sergio_syn/ecoli_add",],
        # "--data_level": ["dropout_poisson100"],
        "--en_vars": [5],
        "--en_obs": [10],
        "--env_idx": [i * 10 + 9 for i in range(1, 5)],
    }

    launch_tasks(param_option=1,
                 base_cmd=base_cmd,
                 param_dict=param_dict,
                 partition='rtx3090',
                 exclude='nutella,radish',
                 timeout="1-00:00:00",
                 job_name="valid",
                 max_job_num=600)


if __name__ == '__main__':
    run()
