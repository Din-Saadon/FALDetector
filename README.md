# Detecting Photoshopped Faces by Scripting Photoshop



## Dataset:

The dataset used for this project can be found at: [Photoshopped Faces on Kaggle](https://www.kaggle.com/datasets/tbourton/photoshopped-faces)

## Download Pre-trained Model Weights:
```

wget https://www.dropbox.com/s/pby9dhpr6cqziyl/local.pth?dl=0 -O ./weights/local.pth

```

## Install Conda:
```

 curl -fsSLO https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
 bash Miniconda3-latest-Linux-x86_64.sh

```


## Setup:

```
conda env create -f environment.yml
conda activate FALDetector
./install.sh
```

## Training Command:

```
python train.py \
    --original_data_dir <path_to_original_dataset> \
    --modified_data_dir <path_to_modified_dataset> \
    --checkpoint <path_to_checkpoint_model> \
    --batch_size <batch_size_value> \
    --save_checkpoint_dir <directory_to_save_model_checkpoints> \
    --epochs <number_of_epochs> \
    --input_size <input_dimensions_value> \
    --strides_ms <stride_values>

## For additional arguments and usage, run: python train.py --help

```

## Prediction Command:

```
python pred.py \
    --input_data_path <path_to_input_data> \
    --output_directory <path_to_output_directory> \
    --model_checkpoint <path_to_model_checkpoint>

## For additional arguments and usage, run: python pred.py --help

```
