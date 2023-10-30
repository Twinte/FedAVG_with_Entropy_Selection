import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set a random seed for reproducibility
np.random.seed(14)
torch.manual_seed(14)  # Change the seed to the same value used earlier

# Hyperparameters
num_clients = 40  # Number of clients
num_epochs = 1    # Number of local epochs
global_rounds = 100
lr = 0.001         # Learning rate
alpha = 0.1     # Dirichlet distribution parameters
num_classes = 10
non_iid = True

#Initialize a logger
log_folder = "logs"
results_folder = "results"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Create a suffix based on the 'non_iid' variable
non_iid_suffix = "_non_iid" if non_iid else "_iid"
# Generate a unique log file name with date and time
log_file = os.path.join(log_folder, f"training_{time.strftime('%Y%m%d_%H%M%S')}{non_iid_suffix}.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Create a directory for results if it doesn't exist
results_dir = os.path.join(results_folder, f"{time.strftime('%Y%m%d_%H%M%S')}{non_iid_suffix}")
os.makedirs(results_dir, exist_ok=True)

# Define the neural network architecture
class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()

        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, 32, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        # First fully connected layer
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True)
        )
        
        # Output layer (fully connected)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)      # Apply first convolutional layer
        out = self.conv2(out)    # Apply second convolutional layer
        out = torch.flatten(out, 1)  # Flatten the output for fully connected layers
        out = self.fc1(out)      # Apply first fully connected layer
        out = self.fc(out)       # Apply output fully connected layer
        return out
    
# Load FMNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
num_samples_per_class = len(trainset) // num_clients  # Each class has 6000 samples for 10 clients
indices = torch.arange(0, len(trainset))
dataset_labels = trainset.targets
class_indices = [indices[i*num_samples_per_class:(i+1)*num_samples_per_class] for i in range(num_clients)]

# Initialize data structures
client_indices = [[] for _ in range(num_clients)]

# Calculate Dirichlet-distributed proportions
if non_iid:
    min_size = 0
    K = num_classes
    N = len(dataset_labels)

    print("Calculating Dirichlet Distribution...")
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

logging.info('-'*50)
logging.info("Length of each client's dataset:")
for i, indices in enumerate(client_indices):
    logging.info(f"Client {i + 1}: {len(indices)} samples")
logging.info(f"Total samples across all clients: {sum(len(indices) for indices in client_indices)}")

    

# Create Subset DataLoaders for each client
client_loaders = [DataLoader(Subset(trainset, client_indices[i]), batch_size=64, shuffle=True) for i in range(num_clients)]

# Initialize global model
global_model = FedAvgCNN().to(device)

# Calculate and report entropy of client labels before training
entropy_scores_list = []  # List to store entropy scores for each client
for client_idx in range(num_clients):
    client_subset_labels = [trainset[idx][1] for idx in class_indices[client_idx]]
    
    unique_labels, label_counts = np.unique(client_subset_labels, return_counts=True)
    print(unique_labels)
    print(label_counts)
    label_probs = label_counts / np.sum(label_counts)
    print(label_probs)
    
    # Remove zero probabilities to avoid log(0) issues
    label_probs = label_probs[label_probs > 0]
    
    entropy = -np.sum(label_probs * np.log2(label_probs))
    
    entropy_scores_list.append(entropy)
    if non_iid:
        logging.info('-'*50)
        logging.info(f"Client {client_idx+1} - Label Entropy: {entropy:.4f}")

if non_iid:
    # Rank clients based on average entropy scores
    ranked_clients = np.argsort(entropy_scores_list)[::-1][:num_clients]  # Sort and take the top 10 clients
else:
    # IID Data Distribution (randomly select 10 clients)
    ranked_clients = np.random.permutation(num_clients)[:num_clients]
    client_indices = [class_indices[ranked_clients[i]] for i in range(num_clients)]

#print(ranked_clients)
# Calculate the number of clients to select (25% of num_clients)
num_clients_to_select = int(0.25 * num_clients)

# Get the subset of ranked clients to train
selected_clients = ranked_clients[:num_clients_to_select]

# Lists to store metrics for each epoch and each client
all_client_train_losses = []
all_client_train_accuracies = []
all_client_train_auc_scores = []

# Log a description of the process before training
logging.info('-'*50)
logging.info("***** Entropy-Based Selection Mechanism for FL *****")
logging.info(f"Number of Clients: {num_clients}")
logging.info(f"Number of Local Epochs: {num_epochs}")
logging.info(f"Number of Global Rounds: {global_rounds}")
logging.info(f"Learning Rate: {lr}")
logging.info(f"Alpha (Dirichlet Distribution Parameter): {alpha}")
logging.info(f"Number of Classes: {num_classes}")
logging.info(f"Non-iid Data Distribution: {non_iid}")
logging.info(f"Device: {device}")

# Training loop (FedAvg)
for round in range(global_rounds):
    local_models = []  # Store local models of clients
    logging.info(f"Epoch {round+1}/{global_rounds}")

    # Lists to store metrics for each client
    client_train_losses = []
    client_train_accuracies = []
    client_train_auc_scores = []

    for client_idx in selected_clients:  # Loop through only the selected clients
        local_model = FedAvgCNN().to(device)  # Create a local copy of the model
        local_model.load_state_dict(global_model.state_dict())  # Initialize with global model
        local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

        # Create Subset DataLoader for the current client
        client_subset = Subset(trainset, class_indices[client_idx])
        client_loader = DataLoader(client_subset, batch_size=64, shuffle=True)

        logging.info('-'*50)
        logging.info(f"  Training Client No. {client_idx+1}...")
        start_time = time.time() #Record start time"

        client_loss_accumulator = 0.0
        all_targets = []
        all_predictions_probs = []

        for local_epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(client_loader):
                data, target = data.to(device), target.to(device)
                local_optimizer.zero_grad()
                output = local_model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                local_optimizer.step()

                # Accumulate loss for the current client
                client_loss_accumulator += loss.item()

                # Append batch predictions and targets
                all_targets.extend(target.tolist())
                all_predictions_probs.extend(torch.softmax(output, dim=1).tolist())

        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time  # Calculate elapsed time

        # Calculate metrics for the current client
        accuracy = accuracy_score(all_targets, torch.argmax(torch.tensor(all_predictions_probs), axis=1))

        # Convert target labels to binary indicators for AUC calculation
        lb = LabelBinarizer()
        lb.fit(all_targets)
        all_targets_binary = lb.transform(all_targets)

        auc_scores = roc_auc_score(all_targets_binary, all_predictions_probs, average='macro')

        # Append metrics to the lists
        client_train_losses.append(client_loss_accumulator / len(client_loader))
        client_train_accuracies.append(accuracy)
        client_train_auc_scores.append(auc_scores)

        local_models.append(local_model)
        # Log information for this client round
        logging.info(f"Client {client_idx+1} - Loss: {client_train_losses[-1]:.4f}, Accuracy: {client_train_accuracies[-1]:.4f}, AUC: {client_train_auc_scores[-1]:.4f}")
        logging.info(f"Client {client_idx+1} - Round Time: {elapsed_time:.2f} seconds")


    # Append metrics for the current epoch
    all_client_train_losses.append(client_train_losses)
    all_client_train_accuracies.append(client_train_accuracies)
    all_client_train_auc_scores.append(client_train_auc_scores)

    # Print metrics for each client
    for i, client_idx in enumerate(selected_clients):
        logging.info('-'*50)
        logging.info(f"Client {client_idx+1} - Loss: {client_train_losses[i]:.4f}, Accuracy: {client_train_accuracies[i]:.4f}, AUC: {client_train_auc_scores[i]:.4f}")

    # Aggregation (FedAvg)
    aggregated_state_dict = {}
    logging.info('-'*50)
    logging.info(f"Aggregating client models...")
    for param_name in global_model.state_dict():
        param_tensors = [local_models[i].state_dict()[param_name] for i in range(len(selected_clients))]
        aggregated_param = torch.mean(torch.stack(param_tensors), dim=0)
        aggregated_state_dict[param_name] = aggregated_param

    global_model.load_state_dict(aggregated_state_dict)

# Close the logger
logging.shutdown()

# Return the final global model
final_global_model = global_model.to("cpu")

# Calculate mean values for each epoch across all clients
mean_train_losses = np.mean(all_client_train_losses, axis=1)
mean_train_accuracies = np.mean(all_client_train_accuracies, axis=1)
mean_train_auc_scores = np.mean(all_client_train_auc_scores, axis=1)

# Plot metrics after the process is completed

# Create the first figure for the loss plot
fig1, ax1 = plt.subplots(figsize=(12, 6))
plt.plot(mean_train_losses)
plt.title('Mean Client Train Loss')
plt.xlabel('Round')
plt.ylabel('Loss')

# Save the first figure in the results directory
fig1.savefig(os.path.join(results_dir, 'mean_train_loss.png'))

# Create the second figure for the accuracy plot
fig2, ax2 = plt.subplots(figsize=(12, 6))
plt.plot(mean_train_accuracies)
plt.title('Mean Client Train Accuracy')
plt.xlabel('Round')
plt.ylabel('Accuracy')

# Save the second figure in the results directory
fig2.savefig(os.path.join(results_dir, 'mean_train_accuracy.png'))

# Create the third figure for the AUC plot
fig3, ax3 = plt.subplots(figsize=(12, 6))
plt.plot(mean_train_auc_scores)
plt.title('Mean Client Train AUC')
plt.xlabel('Round')
plt.ylabel('AUC Score')

# Save the third figure in the results directory
fig3.savefig(os.path.join(results_dir, 'mean_train_auc.png'))

# Visualize class distribution in each client
for client_idx in selected_clients:
    client_subset_labels = [trainset[idx][1] for idx in class_indices[client_idx]]
    
    unique_labels, label_counts = np.unique(client_subset_labels, return_counts=True)
    
    # Create a bar plot for the class distribution
    plt.figure(figsize=(8, 4))
    sns.barplot(x=unique_labels, y=label_counts)
    plt.title(f'Client {client_idx+1} - Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Save the bar plot in the results directory
    plot_name = f'client_{client_idx+1}_class_distribution.png'
    plt.savefig(os.path.join(results_dir, plot_name))
    
    plt.show()