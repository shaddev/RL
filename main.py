import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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

    states = torch.tensor(df[observation_cols].values, dtype=torch.float32)
    actions = torch.tensor(df["a:action"].values, dtype=torch.int64)

    X_train, X_test, y_train, y_test = train_test_split(states, actions, test_size=0.2, random_state=42)

    model = torch.load("mlp.pth")
    model.eval()
    y = torch.argmax(model(states), dim=-1)
    print(torch.sum(y == actions) / len(actions))
    print(y[:200])




def ppo_eval():

    # X_train, X_test, y_train, y_test = train_test_split(states, actions, test_size=0.2, random_state=42)

    # KNN for behavior policy
    # knn = KNeighborsClassifier(n_neighbors=5)
    # knn.fit(X_train, y_train)

    C1 = -0.125
    C2 = -2
    terminal_rewards = torch.tensor(df["r:reward"].values, dtype=torch.float32)
    dones = torch.tensor(df["r:reward"].values != 0, dtype=torch.float32)
    r_states = torch.tensor(df[["o:SOFA","o:Arterial_lactate", "traj", "step"]].values, dtype=torch.float32)
    next_states = torch.cat((torch.zeros(1, 4), r_states[:-1]))
    rewards = C1 * (next_states[:, 0] - r_states[:, 0]) + C2 * np.tanh(next_states[:, 1] - r_states[:, 1])
    rewards[dones == 1] = 15 * terminal_rewards[dones == 1]
    
    mlp = torch.load("mlp.pth").eval()
    ppo_actor = torch.load("ppo_actor.pth").eval()

    mlp_out = mlp(states)
    ppo_actor_out = ppo_actor(states)

    mlp_probs = F.softmax(mlp_out, dim=1)
    mlp_probs = mlp_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    ppo_actor_probs = F.softmax(ppo_actor_out, dim=1)
    ppo_actor_probs = ppo_actor_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    print(mlp_probs.shape)

    rho_all = (ppo_actor_probs / (mlp_probs + 1e-8)).detach().numpy()

    max_timesteps = df['step'].max() + 1

    rho = []
    all_rewards = []
    cur_step = 0
    for traj_id, group in df.groupby('traj'):

        rew = rewards[cur_step : cur_step + len(group)]
        rew = np.pad(rew, (0, max_timesteps - len(group)), mode='constant')
        all_rewards.append(rew)

        rho_group = rho_all[cur_step : cur_step + len(group)]
        rho_group = np.pad(rho_group, (0, max_timesteps - len(group)), mode='constant')
        rho.append(rho_group)

        cur_step += len(group)
    
    rho = np.stack(rho)
    all_rewards = np.stack(all_rewards)

    # print(mlp_probs)



    # # Compute cumulative product of importance weights per time step
    w = torch.cumprod(torch.tensor(rho), dim=1)  # shape [N, H]

    # # Weighted sum per time step
    numerator = (w * all_rewards).sum(dim=0)     # shape [H]
    denominator = w.sum(dim=0) + 1e-8        # avoid div-by-zero

    phwis = (numerator / denominator).sum()  # scalar estimate

    print(f"final score = {phwis}")

    


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
    C1 = -0.125
    C2 = -2
    gamma = 0.9
    epsilon = 0.2
    lr = 0.0001
    weight_decay = 0.00001

    lstm = lstm_train(True)
    actor = mlp_train(True)
    critic = MLP(input_dim, hidden_dim, 1)
    actor_optim = optim.Adam(actor.parameters(), lr=lr, weight_decay=weight_decay)
    critic_optim = optim.Adam(critic.parameters(), lr=lr, weight_decay=weight_decay)

    lstm.eval()
    #actor_model.eval()

    x, y = get_sequences(3, observation_action_cols)

    sofa_index = observation_cols.index('o:SOFA')
    lactate_index = observation_cols.index('o:Arterial_lactate')

    epochs = 250
    trajs_per_epoch = 500
    timesteps = 10
    num_iters = 5
    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1} out of {epochs}")

        batch_rewards = []
        batch_action_probs = []

        # starting state of rollout
        indices = torch.randperm(x.size(0))[:trajs_per_epoch]
        trajs = x[indices]

        # simulate rollout
        for timestep in range(timesteps):

            #print(f"timestep {timestep}")

            # get next traj
            current_timestep = trajs[:,-1,:-1] # get all observation features (i.e. everything except action)
            actor_out = actor(current_timestep)
            actions = torch.argmax(actor_out, dim=-1).unsqueeze(1)
            next_timestep = lstm(trajs[:, [timestep, timestep+1, timestep+2], :]).detach()
            next_timestep = torch.cat([next_timestep, actions], dim=1).unsqueeze(1)
            trajs = torch.cat([trajs, next_timestep], dim=1)

            # log action probs
            probs = F.log_softmax(actor_out, dim=1)
            action_probs = probs.gather(1, actions).squeeze(1)
            batch_action_probs.append(action_probs)

            # print(current_timestep.shape)
            # print(next_timestep.shape)

            reward = (C1 * (next_timestep[:, 0, sofa_index] - current_timestep[:, sofa_index]) \
                    + C2 * np.tanh(next_timestep[:, 0, lactate_index] - current_timestep[:, lactate_index]))
            
            batch_rewards.append(reward)
        #print(torch.stack(batch_action_probs).T.shape)
        
        #batch_action_probs = torch.cat(batch_action_probs, dim=0)
        #batch_rewards = torch.cat(batch_rewards, dim=0)
        #batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32)
        #print(trajs.shape)
        #print(batch_action_probs.shape)
        #print(batch_rewards.shape)
        #print(batch_obs.shape)
        #trajs = trajs.reshape(-1, len(observation_action_cols))
        #batch_rewards = batch_rewards.reshape()

        # flatten so that we have [batch_size * timesteps, probs/rewards/features]
        # fill in the first dimension batch-wise
        batch_action_probs = torch.stack(batch_action_probs).T.reshape(-1).detach()
        batch_rewards = torch.stack(batch_rewards).T.reshape(-1).detach()
        batch_obs = trajs[:, 2:-1, :-1].reshape(-1, len(observation_cols)).detach()

        #with torch.no_grad():
        #V = critic(batch_obs).squeeze() # value
        #A = (batch_rewards - V).detach() # advantage
        A = (batch_rewards - critic(batch_obs).squeeze()).detach()

        critic_criterion = nn.MSELoss()
            
        for i in range(num_iters):

            V = critic(batch_obs).squeeze()
            
            actor_out = actor(batch_obs)
            actions = torch.argmax(actor_out, dim=-1).unsqueeze(-1)
            probs = F.log_softmax(actor_out, dim=1)
            action_probs = probs.gather(1, actions).squeeze(1)

            ratios = torch.exp(action_probs - batch_action_probs)
            surr1 = ratios * A
            surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * A
            
            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = critic_criterion(V, batch_rewards)

            # Calculate gradients and perform backward propagation for actor & critic
            actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()


            print(f"Iteration {i+1}/{num_iters} of epoch {epoch+1}, actor loss: {actor_loss}, critic loss: {critic_loss}")
    
    torch.save(actor, "ppo_actor.pth")

def main():
    #mlp_train()
    #lstm_train()
    #ppo_train()
    #mlp_eval()
    ppo_eval()

main()
    
