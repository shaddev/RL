import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class EnvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(EnvLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.relu = nn.ReLU()
        self.lin = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        x = self.relu(out[:, -1, :])
        x = self.lin(x)
        return x

df = pd.read_csv('../../data.csv')

observation_cols = ['o:gender', 'o:mechvent', 'o:max_dose_vaso', 'o:re_admission', 'o:age',
 'o:Weight_kg', 'o:GCS', 'o:HR', 'o:SysBP', 'o:MeanBP', 'o:DiaBP', 'o:RR', 'o:Temp_C',
'o:FiO2_1', 'o:Potassium', 'o:Sodium', 'o:Chloride', 'o:Glucose', 'o:Magnesium', 'o:Calcium',
'o:Hb', 'o:WBC_count', 'o:Platelets_count', 'o:PTT', 'o:PT', 'o:Arterial_pH', 'o:paO2', 'o:paCO2',
'o:Arterial_BE', 'o:HCO3', 'o:Arterial_lactate', 'o:SOFA', 'o:SIRS', 'o:Shock_Index', 'o:PaO2_FiO2',
 'o:cumulated_balance', 'o:SpO2', 'o:BUN', 'o:Creatinine', 'o:SGOT', 'o:SGPT', 'o:Total_bili', 'o:INR',
 'o:input_total', 'o:input_4hourly', 'o:output_total', 'o:output_4hourly']

observation_action_cols = observation_cols + ["a:action"]

input_dim = len(observation_cols)
hidden_dim = 64
output_dim = np.max(df["a:action"]) + 1

states = torch.tensor(df[observation_cols].values, dtype=torch.float32)
actions = torch.tensor(df["a:action"].values, dtype=torch.int64)

def get_sequences(seq_length, x_cols):
    sequences = []
    y = []
    for traj_id, traj_group in df.groupby('traj'):
        group = traj_group[x_cols].to_numpy()
        if len(group) < seq_length + 1:
            continue  # Skip short trajectories

        for i in range(len(group) - (seq_length)):  # sliding window of size SEQ_LENGTH
            seq = group[i:i+seq_length]  # shape: (SEQ_LENGTH, feature_dim)
            sequences.append(seq)
            y.append(traj_group[observation_cols].to_numpy()[i+seq_length])
    
    x = torch.tensor(np.stack(sequences), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return x, y

def mlp_train(use_saved = False):
    if use_saved:
        return torch.load("mlp.pth")

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
    
    torch.save(model, "mlp.pth")
    return model

def mlp_eval():

    model.eval()
    y = torch.argmax(model(X_test), dim=-1)
    print(torch.sum(y == y_test) / len(y_test))
    print(y[:200])


def lstm_train(use_saved = False):

    if use_saved:
        return torch.load("lstm.pth")

    SEQ_LENGTH = 3

    x,y = get_sequences(SEQ_LENGTH, observation_action_cols)

    input_dim = len(observation_action_cols)
    output_dim = len(observation_cols)
    model = EnvLSTM(input_dim, hidden_dim, SEQ_LENGTH, output_dim)

    lr = 0.01
    weight_decay = 0.00001
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.MSELoss()

    epochs = 1000
    for epoch in range(epochs):
        output = model(x)
        #print(output.dtype)
        #print(output.shape)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    torch.save(model, "lstm.pth")
    return model

def ppo_train():
    C1 = -1
    C2 = -2
    rewards = []

    lstm = lstm_train(True)
    actor_model = mlp_train(True)
    critic_model = MLP(input_dim, hidden_dim, 1)

    lstm.eval()
    #actor_model.eval()

    x, y = get_sequences(3, observation_action_cols)

    sofa_index = observation_cols.index('o:SOFA')
    lactate_index = observation_cols.index('o:Arterial_lactate')

    epochs = 10
    timesteps = 10
    for epoch in range(epochs):
        indices = torch.randperm(x.size(0))[:100]
        trajs = x[indices]

        for timestep in range(timesteps):

            print(f"timestep {timestep}")

            current_timestep = trajs[:,-1,:-1]

            actor_out = actor_model(current_timestep)
            actions = torch.argmax(actor_out, dim=-1).unsqueeze(1)
            next_timestep = lstm(trajs[:, [timestep, timestep+1, timestep+2], :])
            next_timestep = torch.cat([next_timestep, actions], dim=1).unsqueeze(1)
            trajs = torch.cat([trajs, next_timestep], dim=1)

            F.softmax(actor_out, dim=1)

            print(current_timestep.shape)
            print(next_timestep.shape)

            reward = C1 * (next_timestep[:, 0, sofa_index] - current_timestep[:, sofa_index]) \
                    + C2 * (next_timestep[:, 0, lactate_index] - current_timestep[:, lactate_index])
            
            rewards.append(reward)

def main():
    #mlp_train()
    #lstm_train()
    ppo_train()

main()
    
