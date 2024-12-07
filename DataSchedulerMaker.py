import torch
import numpy as np
import pickle

# Load in Datasets
MNIST_coreset = torch.load('MNISTFull_Coreset.pt')
MNIST_odds = torch.load('odd_Dataset_train.pt')

# Convert data back to tensors
coreset_data = MNIST_coreset.tensors[0]
odds_data = MNIST_odds.tensors[0]

coreset_target = MNIST_coreset.targets
odds_target = MNIST_odds.targets

# Begin scheduler
epochs = 100

# Constant mix
constant_mix_data = torch.cat([coreset_data,odds_data], 0)
constant_mix_targets = torch.cat([coreset_target, odds_target], 0)

constant_mix_dataset = torch.utils.data.TensorDataset(constant_mix_data)
constant_mix_dataset.targets = constant_mix_targets

constant_mix_dataloader = [torch.utils.data.DataLoader(constant_mix_dataset, batch_size = 64, shuffle=True)]

with open('MNISTConstantDL.pkl', 'wb') as file:
    pickle.dump(constant_mix_dataloader, file)
 
# Backloaded mixing
mix_ratio = torch.linspace(0,len(constant_mix_dataloader), epochs, dtype=torch.int)

dl_back_list = []
for i in range(epochs):
    mix_indices = torch.randint(0, len(constant_mix_dataloader), [mix_ratio[i].tolist()])

    epoch_data = torch.cat([coreset_data[mix_indices,:,:],odds_data], 0)
    epoch_targets = torch.cat([coreset_target[mix_indices],odds_target],0)

    epoch_dataset = torch.utils.data.TensorDataset(epoch_data)
    epoch_dataset.targets = epoch_targets

    dl_back_list.append(torch.utils.data.DataLoader(epoch_dataset, batch_size = 64, shuffle=True))

with open('MNISTBackLoadDL.pkl', 'wb') as file:
    pickle.dump(dl_back_list, file)

# Frontloaded mixing
mix_ratio = torch.linspace(len(constant_mix_dataloader), 0, epochs, dtype=torch.int)
mix_ratio = torch.flip(mix_ratio,[0])

dl_front_list = []
for i in range(epochs):
    mix_indices = torch.randint(0, len(constant_mix_dataloader), [mix_ratio[i].tolist()])

    epoch_data = torch.cat([coreset_data[mix_indices,:,:],odds_data], 0)
    epoch_targets = torch.cat([coreset_target[mix_indices],odds_target],0)

    epoch_dataset = torch.utils.data.TensorDataset(epoch_data)
    epoch_dataset.targets = epoch_targets

    dl_front_list.append(torch.utils.data.DataLoader(epoch_dataset, batch_size = 64, shuffle=True))

with open('MNISTFrontLoadDL.pkl', 'wb') as file:
    pickle.dump(dl_front_list, file)