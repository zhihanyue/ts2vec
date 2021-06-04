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

To train and evaluate on a dataset, run the following command:

```train & evaluate
python train.py <dataset_name> <run_name> --archive <archive> --batch-size <batch_size> --repr-dims <repr_dims> --gpu <gpu> --eval
```
The detailed descriptions about the arguments are as following:
| Parameter name | Description of parameter |
| --- | --- |
| dataset_name | The dataset name |
| run_name | The folder name used to save model, output and evaluation metrics. This can be set to any word |
| archive | The archive name that the dataset belongs to. This can be set to `UCR`, `UEA`, `forecast_csv`, or `forecast_csv_univar` |
| batch_size | The batch size (defaults to 8) |
| repr_dims | The representation dimensions (defaults to 320) |
| gpu | The gpu no. used for training and inference (defaults to 0) |
| eval | Whether to perform evaluation after training |

(For descriptions of more arguments, run `python train.py -h`.)

After running the above command, the trained encoder, output and evaluation metrics can be found in `training/DatasetName__RunName_Date_Time/`. 

**Scripts:** The scripts for reproduction are provided in `scripts/` folder.


## Code Example

```python
from ts2vec import TS2Vec
import datasets

# Load StarLightCurves dataset from UCR archive
train_data, train_labels, test_data, test_labels = datasets.load_UCR('StarLightCurves')

# Training TS2Vec
model = TS2Vec(
    input_dims=1,
    device=0,
    output_dims=320
)
loss_log = model.fit(
    train_data,
    verbose=True
)

# Obtain learned representations for test set
test_repr = model.encode(test_data)

# Sliding inference for test set
test_repr = model.encode(test_data, casual=True, sliding_padding=100)
    # the timestamp t's representation vector is obtained using the data located in [t-99, t]
```
