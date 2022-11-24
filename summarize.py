from helper import create_summary

# dataset to load
# dataset_name = "drag_simple_steps"
# dataset_name = "pend_simple"
dataset_name = "drag_complex"

# define experiment identifiers
descripor = "alpha"
version = "3"

# create full name for folder containing experiment
experiment_name = f"{dataset_name}_{descripor}_{version}"

create_summary(experiment_name)
