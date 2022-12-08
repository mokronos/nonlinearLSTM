from helper import create_summary

# dataset to load
# --------
# dataset_name = "pend_simple"
# dataset_name = "drag_complex"
# --------
dataset_name = "pend_simple_var"
# dataset_name = "pend_complex"
# dataset_name = "drag_simple_steps"
# dataset_name = "drag_complex_var"
# --------

dataset_names = ["pend_simple_var", "pend_complex", "drag_simple_steps",  "drag_complex_var"]
# dataset_names = ["pend_simple_var"]

# define experiment identifiers
descripor = "alpha"
# version = "2"
versions = ["1", "2", "3", "4"]
# versions = ["3"]
for dataset_name in dataset_names:
    for version in versions:

        # create full name for folder containing experiment
        experiment_name = f"{dataset_name}_{descripor}_{version}"

        create_summary(experiment_name)
