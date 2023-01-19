# nonlinearLSTM

Differential equations are essential for describing the behavior of nonlinear dy-
namic systems over time. But ordinary methods cannot calculate the solutions if
the analytical form is unknown. This work shows the ability of long-short-term-
memory networks (LSTMs) to predict nonlinear dynamic systems over time. Time
series datasets are generated from differential equations of a force pushing an ob-
ject with drag and a pendulum with an external force and friction. The models
are then trained with multiple hyperparameters and training difficulties are high-
lighted. The best models are subsequently evaluated and shown to perform well.
The methods show potential to be used with experimentally obtained data from
differential equations with unknown analytical forms.

This repository contains the code for the work with the above abstract.

# Folders and files
This is just a overview of the different files and folders in this repository.
## Folders
- data: contains the data used for training and testing
- models: contains the trained models and plots of the training/hyperparameter search process
- figures: contains the generated figures for the different datasets and model results
- odes: contains the ODEs used for the different datasets
- results: contains the results as csv of the different models
- test: should contain unit tests, but is currently empty
- ray_results: contains the results of the hyperparameter search with raytune, but not fully working (isn't tracked right now, but will be created if script is ran)
- latex: contains the latex files for the report

## Files
- autotune.py: script for hyperparameter search with raytune, gpu not working for some reason, need to still evaluate trained models
- build_dataset.py: script for building the datasets:
    - splitting into train, validation and test sets
    - scaling the data (Min-Max scaling) and saving scaler for inverse scale later
    - saving the datasets as csv files
- data.py: generator object for creating series with an ODE 
- gen_x scripts: scripts for generating the data for the different datasets; settings are defined in the config json, which then generates input with the generator from data.py
    - gen_template: can be used as template for other datasets
    - gen_drag: generates data for the drag model
    - gen_pend: generates data for the pendulum model
    - gen_getriebe: generates data for the gear model
    - gen_thermal: generates data for the thermal model
- helper.py: helper functions for the different scripts
    - transformations of names
    - dataset creations (cutting series into correct format)
    - dataset splitting
    - loading/writing different files
    - generation of different input shapes (those defined in the config json)
    - data manipulation (mostly multiindexing of pandas dataframes)
    - some sorting and summary creating for the results
- models.py: contains different pytorch lstm models, currently only different amount of layers (probably changed so that the depth is a variable)
- predict_best.py: loads the best model of the specified dataset and experiment, runs the model on train/val/test set and saves results in /results
- search_hyper.py: takes the training/test functions from train.py and runs the hyperparameter search for the specified parameters for a dataset
- train.py: contains training and testing loops ready for manual and raytune hyperparameter search
- summarize.py: takes summarize function from helper.py and loops over all the models to summarize the results and create plots of training progress
- vis.py: uses the functions from vis_helper.py to create the plots for the different datasets and models
- vis_helper.py: contains the functions for creating the plots for the different datasets and models (this started out really general, but is now pretty specific for datasets with 1 or 2 variables)

# Pipeline:

The steps to get a model off the ground are the following:
1. Create a config json for the dataset (see gen_template or the other gen_x files for examples)
    - define the ODE
    - define the inputs of the ODE through the config file (need to probably look into helper functions for creating the inputs)
2. Run the corresponding gen file to generate the data, build_dataset.py gets automatically called
3. Optional but recommended: run vis.py for only the data (not the results, just comment results part out) to create plots for the dataset
4. Run search_hyper.py to run the hyperparameter search for the specified parameters with the dataset
    - define all the parameters you want to search in lists, and the names of the experiment
    - I always called it version 1 for one search, then took the best results, and searched other parameters as version 2
    - currently only lr, nodes, layers working
    - for other parameters search_hyper.py, train.py and probably the models need to be adjusted
    - summarize and predict best gets called automatically after
5. Optional if you change something on the script after training the model, so you dont have to retrain: Run predict_best.py to load the best model and run it on the train/val/test set (need to adjust name of experiment and dataset)
6. Optional if you change something on the script after training the model, so you dont have to retrain: Run summarize.py to create plots of the training progress and save the results as csv
7. Run vis.py to create plots for the results of the model (this time with the results part uncommented)

Most of these things should work with other datasets with more inputs and outputs, but some things might need to be adjusted. Especially the visualizations.

# Requirements
Im just listing some of the packages I used, but it might be better to just go through the files and look whats necessary. Python 3.8.10 was used but newer and slightly older versions should work too.
- pytorch (for the machine learning, gpu version is preferred, but cpu should work too with no changes)
- matplotlib (for the visualizations)
- numpy (for the data manipulation)
- pandas (for the data manipulation, main carrier for the data)
- scipy (for the odeint() solver)
- joblib (for saving the scaler)
- sklearn (for the MinMaxScaler)
- raytune (only for automatic hyperparameter tuning)
- other packages should all be in the standard python library

# Other questions

The code is not super well documented and in parts I created some non optimal solutions to problems. So if there are questions about the models/architectures, optimizer and other things in the process, I would first recommend to read the nonlinearLSTM_report.pdf file and/or check the code itself. If there are still questions after that feel free to contact me via <sebastian.hirt@fau.de> and I'll try to answer any open questions. I'm freezing/archiving this repository to keep it in the state it was, when finishing the report and presentation. For further work it should be forked.


# some training times on 1x gtx 1060 3gb
- 120 samples, 3000 length, 500 epochs
    - five layer 30 min
    - two layer 15 min 
    - width, lr not really affecting time
    - 5000 epochs, 118 mins (two layers/32 nodes)
# inference times:
- 2 layer 64 nodes, 1000 length:
    - 0.035 s
    - --> for one step: 0.000035 s
    - fps (one step): 1/0.000035 = 28571.428571428572
