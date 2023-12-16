# data_loader.py
import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def load_and_split_dataset(num_classes, num_clients, alpha, non_iid=False):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    try:
        trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    except Exception as e:
        raise RuntimeError(f"Error loading the dataset: {e}")

    num_samples_per_class = len(trainset) // num_clients
    indices = torch.arange(0, len(trainset))
    dataset_labels = trainset.targets
    class_indices = [indices[i * num_samples_per_class:(i + 1) * num_samples_per_class] for i in range(num_clients)]

    # Initialize data structures
    client_indices = [[] for _ in range(num_clients)]

    if non_iid:
        min_size = 0
        K = num_classes
        N = len(dataset_labels)

        while min_size < num_classes:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_labels == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])  
        # Assign data samples to clients
        for j in range(num_clients):
            client_indices[j] = idx_batch[j]
    else:
        print("IID partition")
        num_samples_per_client = len(trainset) // num_clients  # Each client has an equal number of samples
        #Shuffle the dataset to create an IID partition
        indices = torch.randperm(len(trainset))
        # Divide the dataset into equal-sized subsets for each client
        client_indices = [indices[i * num_samples_per_client: (i + 1) * num_samples_per_client] for i in range(num_clients)]

    return trainset, client_indices, class_indices

def create_client_loaders(trainset, client_indices):
    client_loaders = [DataLoader(Subset(trainset, client_indices[i]), batch_size=64, shuffle=True) for i in range(len(client_indices))]
    return client_loaders
