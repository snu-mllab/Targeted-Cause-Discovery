from slurm_launcher.sbatch_launcher import launch_tasks
import argparse


def run(args):
    gpu_option = 1
    base_cmd = "python -B src/inference.py --cause --no_tqdm "

    checkpoint = [
        # shift (10)
        # "./results/slurm_gene_dropout_poisson1_sum3_500/model_best_step18800_epoch93_0.159.ckpt",
        # "./results/slurm_gene_data_t2_knn10_norm_sum3_500/model_best_step21600_epoch107_0.155.ckpt",
        # shift (12)
        "./results/slurm_gene_dropout_shift/model_best_step38000_epoch189_0.125-v1.ckpt",
    ]
    names = ["dropout"]

    base_cmd += "--save_pred "
    for i in range(len(names)):
        param_dict1 = {
            "--anchor_type": ["pred"],
            "--anchor_idx": [0],
            "--anchor_size": [20],
            "-c": [checkpoint[0]],
            # Data
            "--shift": [10],
            "--n_env": [100],
            "--data_level": [names[i]],
            "--num_vars": [200],
            "--obs_size": [200],
            "--obs_ratio": [0.05],
            # Ensemble
            "--en_vars": [5],
            "--en_obs": [10],
            # Hyperparams
            "-l": [10],
            "--embed_dim": [16],
            "--n_head": [16],
            "--ffn_embed_dim": [96],
        }

        param_dict_list = [param_dict1]
        job_name_list = [f"anchor_save2"]

        for job_name, param_dict in zip(job_name_list, param_dict_list):
            launch_tasks(param_option=gpu_option,
                         base_cmd=base_cmd,
                         param_dict=param_dict,
                         partition='rtx3090',
                         exclude='kiwi,nutella,alan,ohm,carl,euclid,hamming',
                         timeout="7-00:00:00",
                         job_name=job_name,
                         max_job_num=200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    run(args)
