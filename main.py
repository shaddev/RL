import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

df = pd.read_csv('../../data.csv')

observation_cols = ['o:gender', 'o:mechvent', 'o:max_dose_vaso', 'o:re_admission', 'o:age',
 'o:Weight_kg', 'o:GCS', 'o:HR', 'o:SysBP', 'o:MeanBP', 'o:DiaBP', 'o:RR', 'o:Temp_C',
'o:FiO2_1', 'o:Potassium', 'o:Sodium', 'o:Chloride', 'o:Glucose', 'o:Magnesium', 'o:Calcium',
'o:Hb', 'o:WBC_count', 'o:Platelets_count', 'o:PTT', 'o:PT', 'o:Arterial_pH', 'o:paO2', 'o:paCO2',
'o:Arterial_BE', 'o:HCO3', 'o:Arterial_lactate', 'o:SOFA', 'o:SIRS', 'o:Shock_Index', 'o:PaO2_FiO2',
 'o:cumulated_balance', 'o:SpO2', 'o:BUN', 'o:Creatinine', 'o:SGOT', 'o:SGPT', 'o:Total_bili', 'o:INR',
 'o:input_total', 'o:input_4hourly', 'o:output_total', 'o:output_4hourly']

input_dim = len(observation_cols)
hidden_dim = 64
output_dim = np.max(df["a:action"]) + 1

states = torch.tensor(df[observation_cols].values, dtype=torch.float32)
actions = torch.tensor(df["a:action"].values, dtype=torch.int64)

X_train, X_test, y_train, y_test = train_test_split(states, actions, test_size=0.2, random_state=42)

model = MLP(input_dim, hidden_dim, output_dim)

lr = 0.01
weight_decay = 0.00001
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

criterion = nn.CrossEntropyLoss()

epochs = 200
for epoch in range(epochs):
    output = model(X_train)
    #print(output.dtype)
    #print(actions)
    loss = criterion(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

model.eval()
y = torch.argmax(model(X_test), dim=-1)
print(torch.sum(y == y_test) / len(y_test))
print(y[:200])