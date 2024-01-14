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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("# GPU is available")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("# Load dataset")

data_normalized = pd.read_excel("Model(Transformer57)-huanjing.xlsx")
data_normalized = data_normalized.values
X = data_normalized[0:, 1: -1].astype(float)
Y = data_normalized[0:, -1].astype(int)

def mixup_data(xx, yy, alpha=1.0):

    lam = np.random.beta(alpha, alpha)

    batch_size = np.size(xx[:, 0])
    index = torch.randperm(batch_size)

    mixed_xx = lam * xx + (1 - lam) * xx[index, :]

    mixed_yy = lam * yy + (1 - lam) * yy[index]
    return mixed_xx, mixed_yy

X_mixed1, Y_mixed1 = mixup_data(X, Y, alpha=1.0)
Y_mixed_1 = np.round(Y_mixed1, 0)
X_mixed2, Y_mixed2 = mixup_data(X, Y, alpha=1.0)
Y_mixed_2 = np.round(Y_mixed2, decimals=0)
X_mixed3, Y_mixed3 = mixup_data(X, Y, alpha=1.0)
Y_mixed_3 = np.round(Y_mixed3, decimals=0)
X_mixed4, Y_mixed4 = mixup_data(X, Y, alpha=1.0)
Y_mixed_4 = np.round(Y_mixed4, decimals=0)
X_final = np.concatenate((X, X_mixed1, X_mixed2, X_mixed3, X_mixed4), axis=0)
Y_final = np.concatenate((Y, Y_mixed_1, Y_mixed_2, Y_mixed_3, Y_mixed_4))


X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y_final, test_size=0.2)


X_train = torch.tensor(X_train).float().to(device)
y_train = torch.tensor(Y_train).long().to(device)
X_test = torch.tensor(X_test).float().to(device)
y_test = torch.tensor(Y_test).long().to(device)

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

print("Create model")

model = TransformerModel(input_size=10, num_classes=3).to(device)
model.load_state_dict(torch.load('model_params_huanjing_3.pkl'))

print("Define loss and optimizer")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

print("Train model")

num_epochs = 4000
for epoch in range(num_epochs):

    outputs = model(X_train)

    loss = criterion(outputs, y_train)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Test model")

with torch.no_grad():

    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)

    print(f'outputs:{predicted},y_test:{y_test}')

YYY = Y_test.cpu().numpy()
predictedd = predicted.cpu().numpy()

result = confusion_matrix(YYY, predictedd)
print("Confusion Matrix:")
print(result)
result1 = classification_report(YYY, predictedd)
print("Classification Report:", )
print(result1)
result2 = accuracy_score(YYY, predictedd)
print("Accuracy:", result2)

torch.save(model.state_dict(), "model_params_huanjing_11.pkl")