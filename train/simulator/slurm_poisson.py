from slurm_launcher.sbatch_launcher import launch_tasks
import argparse


def run(args):
    base_cmd = "python -B poisson.py "

    for g in ["sf", "er", "sbm", "sf_indirect", "sf_in"]:
        for e in [2, 4, 6]:
            file = f"{g}_e{e}"
            param_dict = {
                "--name": [file],
                "--idx": [0, 1, 2, 3, 4],
            }

            launch_tasks(param_option=1,
                         base_cmd=base_cmd,
                         param_dict=param_dict,
                         partition='rtx3090',
                         exclude='nutella,banana,alan,ohm',
                         timeout="1-00:00:00",
                         job_name="poisson",
                         max_job_num=600)

    for g in ["ecoli", "yeast_1k"]:  # "ecoli", "yeast_1k", "yeast"
        for tag in ["_func", "_func_noise"]:  # "", "_func", "_func_noise"
            file = f"{g}{tag}"
            param_dict = {
                "--name": [file],
                "--idx": [0, 1, 2, 3, 4],
                "--size": [2],
            }

            launch_tasks(param_option=1,
                         base_cmd=base_cmd,
                         param_dict=param_dict,
                         partition='rtx3090',
                         exclude='nutella,banana,alan,ohm',
                         timeout="1-00:00:00",
                         job_name="poisson",
                         max_job_num=600)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    run(args)
