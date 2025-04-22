import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, MultivariateNormal
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
import pickle

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
    
class EnvTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, heads, num_layers, output_size):
        super(EnvTransformer, self).__init__()

        dropout = 0.1

        self.lin = nn.Linear(input_size, hidden_size)

        self.transformer = nn.Transformer(
            d_model = hidden_size,
            nhead = heads,
            num_encoder_layers = num_layers,
            dim_feedforward = 4 * hidden_size,
            dropout = dropout
        )

        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):

        x = self.lin(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, x)
        x = x[-1, :, :]
        x = self.output(x)
        return x

class MixStrategy(nn.Module):

    def __init__(self, mlp, ppo, low = "mlp", med = "ppo", high = "mlp"):
        super(MixStrategy, self).__init__()
        self.mlp = mlp
        self.ppo = ppo

        # define strategy
        self.low = low
        self.med = med
        self.high = high

        self.low_threshold = -0.5
        self.high_threshold = 0.5

        self.sofa_index = observation_cols.index('o:SOFA')
    
    def forward(self, x):

        sofas = x[:, self.sofa_index]
        low_indices = sofas < self.low_threshold
        high_indices = sofas > self.high_threshold
        med_indices = ~ (low_indices | high_indices)

        out = self.mlp(x)
        ppo_out = self.ppo(x)

        if self.low == "ppo":
            out[low_indices] = ppo_out[low_indices]
        if self.med == "ppo":
            out[med_indices] = ppo_out[med_indices]
        if self.high == "ppo":
            out[high_indices] = ppo_out[high_indices]
        
        return out

df = pd.read_csv('../../data_filtered.csv')
df_test = pd.read_csv('../../data_test_filtered.csv')

# Filter out trajs of size 1
# filtered_groups = df.groupby('traj').size() 
# filtered_groups = filtered_groups[filtered_groups == 1].index
# df = df[~df.isin(filtered_groups)]
# df.to_csv('data_filtered.csv', index=False)
# filtered_groups = df_test.groupby('traj').size()
# filtered_groups = filtered_groups[filtered_groups == 1].index
# df_test = df_test[~df_test.isin(filtered_groups)]
# df_test.to_csv('data_test_filtered.csv', index=False)

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

    states = torch.tensor(df[observation_cols].values, dtype=torch.float32)
    actions = torch.tensor(df["a:action"].values, dtype=torch.int64)

    if use_saved:
        return torch.load("mlp.pth")

    #X_train, X_test, y_train, y_test = train_test_split(states, actions, test_size=0.2, random_state=42)

    model = MLP(input_dim, hidden_dim, output_dim)

    lr = 0.01
    weight_decay = 0.00001
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()

    epochs = 200
    for epoch in range(epochs):
        output = model(states)
        #print(output.dtype)
        #print(actions)
        loss = criterion(output, actions)

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


def ppo_eval(model_file, use_mix = False):

    # X_train, X_test, y_train, y_test = train_test_split(states, actions, test_size=0.2, random_state=42)

    C1 = -0.125
    C2 = -2
    states = torch.tensor(df_test[observation_cols].values, dtype=torch.float32)
    actions = torch.tensor(df_test["a:action"].values, dtype=torch.int64)
    terminal_rewards = torch.tensor(df_test["r:reward"].values, dtype=torch.float32)
    
    r_states = torch.tensor(df_test[["o:SOFA","o:Arterial_lactate", "r:reward"]].values, dtype=torch.float32)
    next_states = torch.cat((torch.zeros(1, 3), r_states[:-1]))
    prev_states = torch.cat((r_states[1:], torch.zeros(1, 3)))
    dones = prev_states[:, 2] != 0
    rewards = C1 * (next_states[:, 0] - r_states[:, 0]) + C2 * np.tanh(next_states[:, 1] - r_states[:, 1])
    rewards[dones == 1] = 15 * terminal_rewards[dones == 1]

    N = len(df_test.groupby("traj"))

    # MLP for behavior policy
    mlp = torch.load("mlp.pth")
    # mlp_out = mlp(states)
    # mlp_probs = F.softmax(mlp_out, dim=1)
    # behavior_probs = mlp_probs.gather(1, actions.unsqueeze(1)).squeeze(1).detach().numpy()

    # KNN for behavior policy
    states_train = torch.tensor(df[observation_cols].values, dtype=torch.float32)
    actions_train = torch.tensor(df["a:action"].values, dtype=torch.int64)
    knn = KNeighborsClassifier(n_neighbors=50)
    knn.fit(states_train, actions_train)
    behavior_probs = knn.predict_proba(df_test[observation_cols].values)
    behavior_probs = behavior_probs[np.arange(behavior_probs.shape[0]), actions]

    saved_ppo_actor = torch.load(model_file)

    if use_mix:
        ppo_actor = MixStrategy(mlp, saved_ppo_actor)
    else:
        ppo_actor = saved_ppo_actor
    
    ppo_actor.eval()

    ppo_actor_out = ppo_actor(states)
    ppo_actor_probs = F.softmax(ppo_actor_out, dim=1)
    ppo_actor_probs = ppo_actor_probs.gather(1, actions.unsqueeze(1)).squeeze(1).detach().numpy()

    rho_all = (ppo_actor_probs / (behavior_probs + 1e-8))

    max_timesteps = df_test['step'].max()

    rho = []
    all_rewards = []
    cur_step = 0
    for traj_id, group in df_test.groupby('traj'):

        num_states = len(group) - 1 # all states except terminal one have a reward

        rew = rewards[cur_step : cur_step + num_states]
        rew = np.pad(rew, (0, max_timesteps - num_states), mode='constant')
        all_rewards.append(rew)

        rho_group = rho_all[cur_step : cur_step + num_states]
        rho_group = np.pad(rho_group, (0, max_timesteps - num_states), mode='constant')
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

    phwis = (numerator / denominator).sum() / N  # scalar estimate

    print(f"final score = {phwis}")


def lstm_train(use_saved = False):

    if use_saved:
        return torch.load("lstm.pth")

    SEQ_LENGTH = 3

    x, y = get_sequences(SEQ_LENGTH, observation_action_cols)

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

def transformer_train():

    # if use_saved:
    #     return torch.load("lstm.pth")

    SEQ_LENGTH = 3

    x, y = get_sequences(SEQ_LENGTH, observation_action_cols)

    input_dim = len(observation_action_cols)
    output_dim = len(observation_cols)
    hidden_dim = 64
    num_heads = 4
    num_layers = 3
    model = EnvTransformer(input_dim, hidden_dim, num_heads, num_layers, output_dim)

    lr = 0.001
    weight_decay = 0.00001
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.MSELoss()

    epochs = 200
    for epoch in range(epochs):
        output = model(x)
        #print(output.dtype)
        #print(output.shape)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    torch.save(model, "transformer.pth")
    return model


def ppo_train(env_type = "lstm"):
    C1 = -0.125
    C2 = -2
    gamma = 0.9
    epsilon = 0.2
    lr = 0.0001
    weight_decay = 0.00001

    if env_type == "lstm":
        env_model = lstm_train(True)
    else:
        env_model = torch.load("transformer.pth")
    
    actor = mlp_train(True)
    critic = MLP(input_dim, hidden_dim, 1)
    actor_optim = optim.Adam(actor.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-5)
    critic_optim = optim.Adam(critic.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-5)

    env_model.eval()

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

            dist = Categorical(logits=actor_out)
            actions = dist.sample().unsqueeze(1)

            #actions = torch.argmax(actor_out, dim=-1).unsqueeze(1)
            
            next_timestep = env_model(trajs[:, [timestep, timestep+1, timestep+2], :]).detach()
            # print(actions.shape) # uncomment for lstm
            # print(next_timestep.shape)
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
    
    torch.save(actor, f"ppo_actor_{env_type}.pth")

def fqi():

    # Initialize
    Q_models = {}  # One model per action
    num_iterations = 10
    gamma = 0.99  # discount factor

    C1 = -0.125
    C2 = -2
    states = torch.tensor(df[observation_cols].values, dtype=torch.float32)
    actions = torch.tensor(df["a:action"].values, dtype=torch.int64)

    next_states = torch.cat((torch.zeros(1, len(observation_cols)), states[:-1]))
    r_states = torch.tensor(df[["o:SOFA","o:Arterial_lactate", "r:reward"]].values, dtype=torch.float32)
    prev_states = torch.cat((r_states[1:], torch.zeros(1, 3)))
    terminal_rewards = torch.tensor(df["r:reward"].values, dtype=torch.float32)
    
    terminals = r_states[:, 2] == 1

    dones = prev_states[:, 2] != 0
    rewards = C1 * (next_states[:, 0] - r_states[:, 0]) + C2 * np.tanh(next_states[:, 1] - r_states[:, 1])
    rewards[dones == 1] = 15 * terminal_rewards[dones == 1]

    # states = states[terminals != 1]
    # actions = actions[terminals != 1]
    # rewards = rewards[terminals != 1]

    Q_models = {a: RandomForestRegressor(n_estimators=50) for a in range(25)}

    # Initial Q values = 0
    Q_values = np.zeros(len(states))

    for iteration in range(num_iterations):
        # Target values for training
        targets = []
        features = []
        for a in range(25):
            print(f"Iteration {iteration} Action {a}")
            indices = torch.logical_and(actions == a, terminals != 1)

            if not indices.any():
                raise RuntimeError("no actions")

            s = states[indices]
            s_next = next_states[indices]
            r = rewards[indices]

            # Estimate max_a' Q(s', a') from previous iteration
            max_next_Q = np.zeros(len(s_next))
            for a_prime in range(25):
                if iteration > 0:
                    max_next_Q = np.maximum(
                        max_next_Q,
                        Q_models[a_prime].predict(s_next)
                    )

            y = r + gamma * max_next_Q
            X = s  # current state is input

            # Fit Q_model[a] with (s, y)
            Q_models[a].fit(X, y)
        
    with open('forest.pkl', 'wb') as f:
        pickle.dump(Q_models, f)
    
    return Q_models

def phwdr(model_file, use_mix = False):
    
    C1 = -0.125
    C2 = -2
    states = torch.tensor(df_test[observation_cols].values, dtype=torch.float32)
    actions = torch.tensor(df_test["a:action"].values, dtype=torch.int64)
    terminal_rewards = torch.tensor(df_test["r:reward"].values, dtype=torch.float32)
    
    r_states = torch.tensor(df_test[["o:SOFA","o:Arterial_lactate", "r:reward"]].values, dtype=torch.float32)
    next_states = torch.cat((torch.zeros(1, 3), r_states[:-1]))
    prev_states = torch.cat((r_states[1:], torch.zeros(1, 3)))
    dones = prev_states[:, 2] != 0
    rewards = C1 * (next_states[:, 0] - r_states[:, 0]) + C2 * np.tanh(next_states[:, 1] - r_states[:, 1])
    rewards[dones == 1] = 15 * terminal_rewards[dones == 1]

    N = len(df_test.groupby("traj"))

    # Random Forests for behavior policy
    with open('forest.pkl', 'rb') as f:
        q_models = pickle.load(f)
    
    behavior_preds = np.array([q_models[a].predict(states) for a in range(25)]).T

    behavior_probs = behavior_preds - np.max(behavior_preds, axis=1)[:, np.newaxis]
    exps = np.exp(behavior_probs)
    behavior_probs = exps / np.sum(exps, axis=1)[:, np.newaxis]

    behavior_preds_actions = behavior_preds[np.arange(behavior_preds.shape[0]), actions]
    behavior_probs_actions = behavior_preds[np.arange(behavior_probs.shape[0]), actions]

    print(behavior_probs.shape)
    saved_ppo_actor = torch.load(model_file)

    if use_mix:
        mlp = torch.load("mlp.pth")
        ppo_actor = MixStrategy(mlp, saved_ppo_actor)
    else:
        ppo_actor = saved_ppo_actor
    
    ppo_actor.eval()

    ppo_actor_out = ppo_actor(states)
    ppo_actor_probs = F.softmax(ppo_actor_out, dim=1)
    ppo_actor_probs = ppo_actor_probs.gather(1, actions.unsqueeze(1)).squeeze(1).detach().numpy()

    rho_all = (ppo_actor_probs / (behavior_probs_actions + 1e-8))

    max_timesteps = df_test['step'].max()
    
    all_qsa = []
    rho = []
    all_rewards = []

    all_vs = []
    
    cur_step = 0
    for traj_id, group in df_test.groupby('traj'):

        num_states = len(group) - 1 # all states except terminal one have a reward

        rew = rewards[cur_step : cur_step + num_states]
        rew = np.pad(rew, (0, max_timesteps - num_states), mode='constant')
        all_rewards.append(rew)

        rho_group = rho_all[cur_step : cur_step + num_states]
        rho_group = np.pad(rho_group, (0, max_timesteps - num_states), mode='constant')
        rho.append(rho_group)

        qsa_group = behavior_preds_actions[cur_step : cur_step + num_states]
        qsa_group = np.pad(qsa_group, (0, max_timesteps - num_states), mode='constant')
        all_qsa.append(qsa_group)

        vs_group = [behavior_preds[i] @ behavior_probs[i] for i in range(cur_step, cur_step + num_states + 1)]
        vs_group = np.pad(vs_group, (0, max_timesteps - num_states), mode='constant')
        all_vs.append(vs_group)

        cur_step += len(group)
    
    rho = np.stack(rho)
    all_rewards = np.stack(all_rewards)

    w = torch.cumprod(torch.tensor(rho), dim=1)

    numerator = (w * (all_rewards - all_qsa)).sum(dim=0)
    denominator = w.sum(dim=0) + 1e-8

    sum_1 = (numerator / denominator).sum() / N

    sum_2 = np.sum(vs_group) / N

    score = sum_1 + sum_2

    print(f"phwdr score = {score}")

def split_dataset(max_traj_id = 5000):

    df = pd.read_csv('../../sepsis_final_data_withTimes.csv')

    filtered_groups = df.groupby('traj').size() 
    filtered_groups = filtered_groups[filtered_groups == 1].index
    df = df[~df.isin(filtered_groups)]

    df = df[df['traj'] <= max_traj_id]

    groups = df['traj']
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    train_idx, test_idx = next(gss.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    valid = len(set(train_df['a:action'].unique())) == 25 and len(set(test_df['a:action'].unique())) == 25
    train_trajs = len(train_df['traj'].unique())
    test_trajs = len(test_df['traj'].unique())
    print(f"Train trajs: {train_trajs}, Test trajs: {test_trajs}")
    print(f"Split is valid: {valid}")

    train_df.to_csv('data_filtered_train.csv', index=False)
    test_df.to_csv('data_filtered_test.csv', index=False)


def main():
    #mlp_train()
    #lstm_train()
    #ppo_train("lstm")
    #mlp_eval()
    #ppo_eval("ppo_actor_transformer.pth", True)
    #phwdr("ppo_actor_lstm.pth", True)
    split_dataset()

    #transformer_train()

    #fqi()
    #phwdr()

    # LSTM - final score = 27.956998825073242
    # Transformer - final score = 12.839700698852539
    #               final score = 26.4208927154541

    #print(df.groupby('traj').size().min())

main()
    
