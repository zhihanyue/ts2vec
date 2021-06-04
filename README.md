# TS2Vec

This repository contains the official implementation for the paper "Learning Timestamp-Level Representations for Time Series with Hierarchical Contrastive Loss".

## Requirements

[TODO]

## Data

The datasets can be obtained on the following locations:

[ETT datasets](https://github.com/zhouhaoyi/ETDataset)
[UCR datasets](https://www.cs.ucr.edu/~eamonn/time_series_data_2018)
[UEA datasets](http://www.timeseriesclassification.com)

The datasets should be put into `datasets/` folder.

## Usage

To train and evaluate on some dataset, run the following command:

```train & evaluate
python train.py <dataset_name> <run_name> --archive <archive> --batch-size <batch_size> --repr-dims <repr_dims> --gpu <gpu> --eval
```
The detailed descriptions about the arguments are as following:
| Parameter name | Description of parameter |
| --- | --- |
| dataset_name | The dataset name. |
| run_name | The folder name used to save model, output and evaluation metrics. This can be set to any word. |
| archive | The archive name that the dataset belongs to. This can be set to `UCR`, `UEA`, `forecast_csv`, or `forecast_csv_univar`. |
| batch_size | The batch size, defaults to 8. |
| repr_dims | The representation dimensions, defaults to 320. |
| gpu | The gpu number used for training and inference, defaults to 0. |
| eval | Whether to evaluation after training. |

After running the above command, the model, output and evaluation metrics can be found in `training/DatasetName__RunName_*/`. For more argument descriptions, run `python train.py -h`.

To facilitate reproduction of our results, the scripts for experiments are provided in `scripts/` folder.

## Code Examples

[TODO]



