import torch
import pandas as pd
from torch import nn
import numpy as np
import torch.nn.functional as F
from sklearn import datasets
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data_normalized = pd.read_excel("Model(Transformer9)-huanjing_prediction.xlsx")
data_normalized = data_normalized.values
X = data_normalized[0:, 1:].astype(float)

X_train = torch.tensor(X).float().to(device)

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TransformerModel, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=5)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.encoder(x)

        x = x.squeeze(1)

        x = self.fc(x)
        return x


model = TransformerModel(input_size=10, num_classes=3).to(device)
model.load_state_dict(torch.load('model_params_huanjing_11.pkl'))


with torch.no_grad():

    outputs = model(X_train)
    _, predicted = torch.max(outputs.data, 1)

    print(f'outputs:{predicted}')
