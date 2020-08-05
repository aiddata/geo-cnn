# Settings JSON README
The settings JSON defines all of the variables which control your jobs. This doc will guide you through the purposes of various options that can be set and how they interact with one another. Each section below will refer to a top level block (i.e., key value pair) of options within the settings JSON. Each block is a dict containing a specific group of settings.


### Config
The config block contains high level options about how your environment is set up and which portions of the available training/prediction functions will be run.

- base_path: The root of your project directory. All data must exist within this directory, as subsequent data paths are generated automatically relative to the base_path
- mode: Defines whether parameters for training a CNN are generated based on "batch" block or the "csv" block. Both of these are top level blocks with their own settings discussed in later sections of this doc. The "batch" option allows user to set a range of parameters to create a grid search for training or prediction functions. The "csv" option allows user to specify previously run (trained) hashes representing hyperparameters that were defined using a "batch" from a previous run (I have never had a need for this, and it has not been tested recently.)
- predict: Specifies whether the data used (CSV) for prediction comes from an existing survey ("survey_predict") or a custom set of locations ("source_predict"). Locations should be specified by "lon" and "lat" columns.

Version tags must be manually adjusted by the user. They can be used to distinguish between various groups of tests, changes in code, or anything else the user whiches to classify as different from existing runs. Tags are included in filenames at various stages as described below. Filenames also include unique hashes representing a specific set of training/prediction/other parameters, which ensure that every combination of version tag and parameters is unique. All tags should be only alphanumeric characters. A suggested naming scheme is given as examples below, but it is up to user to choose what works.
- version: Used during data prep and training (example: "v25")
- predict_tag: Used during prediction (example: "p25")
- model_tag: Used for second stage models (example: "m25")
- surface_tag: Used when generating surfaces (example: "s25")

- quiet: true/false, will quiet the output of some functions (primarily output from individual epochs while training)
- overwrite_sample_prep: true/false, whether to overwrite the output from the cnn prep stage
- overwrite_train: true/false, whether to overwrite the output from the cnn traning stage
- overwrite_predict: true/false, whether to overwrite the output from the cnn prediction stage
- run: {
    "train": true/false, whether to run specified cnn stage
    "test": true/false, whether to run specified cnn stage
    "predict": true/false, whether to run specified cnn stage
    "custom_predict": true/false, whether to run specified cnn stage
},
- cuda_device_id: id for cuda device (gpu). Only needed if multiple GPUs are available. Generally just leave as 0
- second_stage_mode: "parallel" or "serial", whether to run second stage models in parallel or serial
- directories: the directories in `<base_path>/outputs` to create


### Static
The static block contains options which are used across multiple stages of processing, but is most heavily tied to data preparation and CNN training.

- imagery_type: imagery id (e.g., "landsat8"). Refers to directory with `<base_path>/landsat/data`
- imagery_bands: bands of imagery to use (corresponds to value in landsat path)

    - source_name: identifier for source used to load samples. Must be CSV file basename within `<base_path>/data/surveys` directory

- sample_definition:
    - imagery: list of imagery temporal identifier to use
    - sample: list

    - sample_type: source/grid/random, defines whether to use a predefined set of samples (source) such as from existing survey locations, or to use a generated grid of sample locations, or to generate random locations (random - not currently functional)

    - random options:
        - random_samples: [Not currently functional] true/false, generate random samples not related to existing sample locations
        - random_count: [Not currently functional] number of random samples to generate
        - random_buffer: [Not currently functional] distance from existing sample locations new random locations must be
        - random_init: [Not currently functional] ?

    - grid options:
        - grid_boundary_file: path of boundary file relative to the `<base_path>/data/boundary` directory. The boundary file should be a GeoJSON that consists of a singly valid polygon which encompases the entire study area. This will be used to define sample and prediction grids.
        - grid_pixel_size: pixel size to use when generating sample grid

    - sample_nfill: number of locations to "fill" or create associated with each existing sample location
    - sample_fill_dist: maximum distance to fill for each sample location
    - sample_fill_mode: fixed/random, whether to fill locations using a fixed grid (regular intervals within sample_fill_dist) or randomly (but still within sample_fill_dist)

- cat_names: list of names to associate with each bin for classifier
- cat_bins: list of values associated with each bin for classifier (lower bound, using `>=` progressing through list)
- cat_field: field name to use for classifying bins in underlying data (field may either exist in source if using source samples, or may be nighttime lights - or other value which a function exists in code - to generate values for source or grid samples. Only NTL function currently exists.)
- ntl_type: dmsp/viirs, which NTL dataset to use for generating NTL values. Must be within temporal bounds of available data.
- ntl_year: year of NTL data to use by default for samples
- ntl_calibrated: whether to use calibrated data or raw when using dmsp NTL
- ntl_dim: dimension of pixel grid square around sample location to use for NTL value. Make sure to adjust based on resolution of NTL data used (500m VIIRS, 1km DMSP)
- ntl_min: drop samples with NTL values below the ntl_min
- type_names: Names of different sample dataframes to generate. Currently may include combination of ["train", "val", "test", "predict"]
- type_weights: Percentage of samples to allocate to each sample dataframe. Must total to 1

### Batch
The batch field is used to perform a grid search during the training of a CNN. It consists of key value pairs where every value is a list of multiple values to be used in a grid search.

- run_type: fine tuning (1) or fixed (0) weights when training the CNN
- n_epochs: number of epochs (1+)
- optim: sgd/adam, optimization algorithm
- lr: learning rate
- momentum: momentum associated with LR
- step_size: every step_size number of epochs LR is adjusted by gamma
- gamma: amount to adjust LR by each step_size epochs (LR * gamma)
- loss_weights: list of weights (size of list must correspond to number of bins for categories in cat_bins). Used to handle imbalanced sampled. Currently just fixed at 1 for all values as we clip class sizes so they are all equal during data prep.
- net: type of network. Currently only includes resnets
- batch_size: Batch size for training
- num_workers: Number of workers to used when training
- dim: dimension of images used. Default 224 for resnet
- agg_method: aggregation method used to generate imagery. Has included mean/min/max when exploring SLC corrections with Landsat7, now just used mean for Landsat8. Used imagery path/file name lookup.

### CSV
The CSV was intended to be used to load specific hash combinations from training for later stage predictions. It was never used beyond initial functionallity testing, but currently is likely not functional. Will probably remove at some point in future.
- path: path to CSV file
- field: field in CSV containing hash value


### source_predict
Settings for generating CNN predictions using source locations (CSV containg lon/lat)

- source: absolute path to source file
- imagery_year: list of imagery temporal identifiers to use (same as in static block used for training)

NTL fields below are the same as in static block and are used to include NTL values with prediction outputs for comparison/analysis.
- ntl_type
- ntl_year
- ntl_calibrated
- ntl_dim


### survey_predict
Settings for generating CNN predictions using locations from survey dataset

To be updated...


### second_stage
Settings for training second stage models.

To be updated...


### third_stage
Settings for running predictions on first and second stage models, and using results to produce surfaces.

__grid__

 To be updated...

__predict__

To be updated...

__surface__

- input_stage: s1/s2, s1 generates surface directly from s1 CNN predictions, s2 using second stage model outputs
- value_type: field from output CSV
- pixel_size: resolution of raster. Manually defined by user so must be sure it matches resolution of grid/data used for predicting underlying surface values or resulting surface will fail or be distorted
- pixel_agg: Not currently used (intended to aggregate pixels to coarser resolution)
- dim: dimensions of imagery used for training/prediction. Used here to evaluate no data in predicted scenes
- scene_max_nodata: Max ratio of no data values in prediction scene
- nodata_val: No data value in imagery