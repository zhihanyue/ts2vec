import os
import pickle 
from datetime import datetime

import numpy as np 
import pandas as pd 
# import matplotlib.pyplot as plt

from ts2vec import TS2Vec
import torch
from torch import nn


# Model traing parameters 
OUTPUT_DIMS = 320 
TEMPORAL_UNIT = 2 
BATCH_SIZE = 128
N_EPOCHS = 2
HIDDEN_DIMS = 64
KERNEL_SIZE = 3
SAVE_CHECK_POINT = False


# load train data 
train_data_dir = "/Users/fguo/cmt/ts2vec/sample_data/train_data_sample10.npy"
train_motion_names_dir = "/Users/fguo/cmt/ts2vec/sample_data/train_motion_names_sample10.parquet"
train_data = np.load(train_data_dir)
train_motion_names = pd.read_parquet(train_motion_names_dir)
print(train_data.shape, train_motion_names.shape)

# load val data 
val_data_dir = "/Users/fguo/cmt/ts2vec/sample_data/val_data_sample4.npy"
val_motion_names_dir = "/Users/fguo/cmt/ts2vec/sample_data/val_motion_names_sample4.parquet"
val_data = np.load(val_data_dir)
val_motion_names = pd.read_parquet(val_motion_names_dir)
print(val_data.shape, val_motion_names.shape)


model = TS2Vec(
    input_dims=train_data.shape[-1],
    device='cpu', 
    output_dims=OUTPUT_DIMS,
    hidden_dims=HIDDEN_DIMS, 
    temporal_unit=TEMPORAL_UNIT,
    batch_size=BATCH_SIZE,
    after_epoch_callback=None
)
# loss_log = model.fit(
#     train_data,
#     val_data,
#     verbose=True, 
#     n_epochs=N_EPOCHS, 
#     )
# model.save("/Users/fguo/cmt/ts2vec/model_checkpoints/sample_model.pkl")

model.load("/Users/fguo/cmt/ts2vec/model_checkpoints/sample_model.pkl")
random_input = torch.randn(1, 1, 8, device="cpu")
# out = model.encode(random_input)


# class MLPModel(nn.Module):
#   def __init__(self):
#       super().__init__()
#       self.fc0 = nn.Linear(8, 8, bias=True)
#       self.fc1 = nn.Linear(8, 4, bias=True)
#       self.fc2 = nn.Linear(4, 2, bias=True)
#       self.fc3 = nn.Linear(2, 2, bias=True)

#   def forward(self, tensor_x: torch.Tensor):
#       tensor_x = self.fc0(tensor_x)
#       tensor_x = torch.sigmoid(tensor_x)
#       tensor_x = self.fc1(tensor_x)
#       tensor_x = torch.sigmoid(tensor_x)
#       tensor_x = self.fc2(tensor_x)
#       tensor_x = torch.sigmoid(tensor_x)
#       output = self.fc3(tensor_x)
#       return output

# model = MLPModel()
# tensor_x = torch.rand((97, 8), dtype=torch.float32)
# onnx_program = torch.onnx.dynamo_export(model, tensor_x)
