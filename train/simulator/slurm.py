from slurm_launcher.sbatch_launcher import launch_tasks

DIR_CONFIG = './experiments/gene-cause'
DIR_OUTPUT = '/storage/janghyun/datasets/causal/train_sergio_syn'


def run():
    base_cmd = "python -B -m avici.experiment.data "

    for g in ["sf", "er", "sbm", "sf_indirect", "sf_in"]:
        for e in [2, 4, 6]:
            file = f"data_{g}_e{e}"
            param_dict = {
                "--j": [i for i in range(50)],
                "--data_config_path": [f"{DIR_CONFIG}/{file}.yaml"],
                "--path_data": [f"{DIR_OUTPUT}/{file[5:]}_add"],
                "--descr": [file],
            }

            launch_tasks(param_option=20,
                         base_cmd=base_cmd,
                         param_dict=param_dict,
                         partition='rtx3090',
                         exclude='nutella,xoi,banana,alan,ohm',
                         timeout="7-00:00:00",
                         job_name="add-" + file[5:],
                         max_job_num=600)

    for g in ["ecoli"]:  # "ecoli", "yeast_1k", "yeast"
        for tag in ["", "_func", "_func_noise"]:
            file = f"data_{g}{tag}"
            param_dict = {
                "--j": [i for i in range(10)],
                "--data_config_path": [f"{DIR_CONFIG}/{file}.yaml"],
                "--path_data": [f"{DIR_OUTPUT}/{file[5:]}_add"],
                "--descr": [file],
            }

            launch_tasks(param_option=25,
                         base_cmd=base_cmd,
                         param_dict=param_dict,
                         partition='rtx3090',
                         exclude='nutella,alan,ohm',
                         timeout="7-00:00:00",
                         job_name=file[5:],
                         max_job_num=600)


if __name__ == '__main__':
    run()
