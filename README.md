# Using Convolutional Neural Networks to Predict Survey Estimates

## Stage 0 - Settings and Data Preparation

* push.sh - shell script to push local data to HPC (script args define dir on HPC and settings JSON to use)
* settings_builder.py - class to manage settings for different scripts
* settings/ - folder containing settings JSONs for jobs
* scripts/ - folder containing one off scripts for preparing survey data and anything else not a core utility used in following stages


## Stage 1 - Training CNN

* s1_jobscripts - jobscript
* main.py - primary script called by jobscript
* data_prep.py - class/functions to generate sample data for training CNN
* create_grid.py - class to generate point grid
* runscript.py - core class/functions for running PyTorch CNN
* load_data.py - classes for reading Landsat imagery for CNN training data and extracting NTL imagery values for training data labels
* load_survey_data.py - class to load survey data for use with CNN predictions
* resnet.py - modified ResNet class (based on PyTorch ResNet class)
* vgg.py - modified VGG class (based on PyTorch VGG class) - NOT FUNCTION


## Stage 2 - Training Secondary Models

* s2_jobscript - jobscript
* second_stage_model.py - primary script called by jobscript
* model_prep.py - functions and classes for building second stage models

* merge_outputs.py - script to merge second stage model metrics


## Stage 3 - Generating Predictive Surface

* build_surface_grid.py

### S3A - Creating Point Grid

### S3B - Generating CNN Features for Surface Grid

### S3C - Generating Secondary Model Values for Surface Grid

### S3D - Building Raster Surface


-------------------------------------------------------------------------------


# Usage Notes


### Stage 1 Training


### Stage 1 Predict


### Stage 2 Train and Predict

- running with full set of cnn features (or more; ntl, etc.) will make stage 2 take significantly longer. If testing, start with PCA instead of full feature set.
- when using MLP Classifier model, single variable models (e.g., only NTL) will not converge

### Stage 3
