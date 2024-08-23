from slurm_launcher.sbatch_launcher import launch_tasks


def run():
    base_cmd = "python -B run.py --save_pred True "

    param_dict = {
        "--data_file": ["human"],
        "--model_type": ["discrete"],
        "--en_vars": [1],
        "--en_obs": [10],
        "--range": [200],
        "--target_idx": [i for i in range(10)],
    }

    launch_tasks(param_option=1,
                 base_cmd=base_cmd,
                 param_dict=param_dict,
                 partition='rtx3090',
                 timeout="1-00:00:00",
                 job_name="test",
                 max_job_num=600)


if __name__ == '__main__':
    run()
