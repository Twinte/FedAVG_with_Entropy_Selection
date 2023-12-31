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
import argparse
import sys

from model import FedAvgCNN
from data_loader import load_and_split_dataset, create_client_loaders
from plotting import plot_mean_metrics, plot_class_distribution

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set a random seed for reproducibility
np.random.seed(14)
torch.manual_seed(14)  # Change the seed to the same value used earlier

parser = argparse.ArgumentParser(description='Federated Learning with FedAvg CNN')

# Add command-line arguments for hyperparameters
parser.add_argument('--num_clients', type=int, default=60, help='Number of clients')
parser.add_argument('--num_epochs', type=int, default=1, help='Number of local epochs')
parser.add_argument('--global_rounds', type=int, default=50, help='Number of global rounds')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--alpha', type=float, default=0.1, help='Dirichlet distribution parameter')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
parser.add_argument('--drop_rate', type=float, default=0.16, help='Client dropout rate')
parser.add_argument('--non_iid', action='store_true', help='Use non-IID data distribution')
parser.add_argument('--entropy_selection', action='store_true', help='Use entropy-based client selection')
parser.add_argument('--join_ratio', type=float, default=1.0, help='Number of clients to select')
parser.add_argument('--pre_trained', action='store_true', help='Use pre-trained model')

args = parser.parse_args()

#Hyperparameters
num_clients = args.num_clients
num_epochs = args.num_epochs
global_rounds = args.global_rounds
lr = args.lr
alpha = args.alpha
num_classes = args.num_classes
drop_rate = args.drop_rate
non_iid = args.non_iid
entropy_selection = args.entropy_selection
num_clients_to_select = int(args.join_ratio*num_clients)
pre_trained = args.pre_trained

# Define the initial number of rounds before dropping clients
initial_rounds_before_drop = 5  # Adjust this as needed
round_to_drop_clients = initial_rounds_before_drop

#Initialize a logger
log_folder = "logs"
results_folder = "results"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Create a suffix based on the 'non_iid' variable and other parameters
non_iid_suffix = "_non_iid" if non_iid else "_iid"
selection_suffix = "_entropy" if entropy_selection else "_random"
drop_rate_suffix = f"_drop{int(drop_rate * 100)}"
# Generate a unique log file name with date and time
log_file = os.path.join(log_folder, f"training_{time.strftime('%Y%m%d_%H%M%S')}{non_iid_suffix}{selection_suffix}{drop_rate_suffix}.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Create a directory for results if it doesn't exist
results_dir = os.path.join(results_folder, f"{time.strftime('%Y%m%d_%H%M%S')}{non_iid_suffix}{selection_suffix}{drop_rate_suffix}")
os.makedirs(results_dir, exist_ok=True)

# Load and split the dataset
trainset, client_indices, class_indices = load_and_split_dataset(num_classes, num_clients, alpha, non_iid)

logging.info('-'*50)
logging.info("Length of each client's dataset:")
for i, indices in enumerate(client_indices):
    logging.info(f"Client {i + 1}: {len(indices)} samples")
logging.info(f"Total samples across all clients: {sum(len(indices) for indices in client_indices)}")

# Create Subset DataLoaders for each client
client_loaders = create_client_loaders(trainset, client_indices)

# Initialize global model
global_model = FedAvgCNN().to(device)

# Load the pre-trained model state dictionary
pretrained_model_path = 'results/global_model.pth'
if os.path.exists(pretrained_model_path) and pre_trained:
    global_model.load_state_dict(torch.load(pretrained_model_path))
    print("Pre-trained model loaded successfully.")
else:
    print("No pre-trained model found. Training from scratch...")

# Initialize a list to store entropy scores for each client
entropy_scores_list = []

for client_idx in range(num_clients):
    client_subset_labels = [trainset[idx][1] for idx in class_indices[client_idx]]

    # Calculate the frequency of each label in the client subset
    label_counts = np.bincount(client_subset_labels)
    
    # Calculate the proportion of each label in the client subset
    label_probs = label_counts / np.sum(label_counts)
    
    # Remove zero probabilities to avoid log(0) issues
    label_probs = label_probs[label_probs > 0]
    
    # Calculate the entropy using the formula you provided
    entropy = -np.sum(label_probs * np.log2(label_probs))
    
    entropy_scores_list.append(entropy)
    if non_iid:
        logging.info('-'*50)
        logging.info(f"Client {client_idx+1} - Label Entropy: {entropy:.4f}")

if non_iid:
    # Rank clients based on average entropy scores
    ranked_clients = np.argsort(entropy_scores_list)[::-1][:num_clients]  # Sort and take the top % of clients
else:
    # IID Data Distribution (randomly select a % of clients)
    ranked_clients = np.random.permutation(num_clients)[:num_clients]
    client_indices = [class_indices[ranked_clients[i]] for i in range(num_clients)]

def select_clients(num_clients, num_clients_to_select, ranked_clients=None, entropy_selection=True):
    if entropy_selection:
        if ranked_clients is None:
            raise ValueError("Ranked clients must be provided for entropy selection.")
        selected_clients = ranked_clients[:num_clients_to_select]
    else:
        selected_clients = np.random.choice(range(num_clients), num_clients_to_select, replace=False)

    return selected_clients

# Lists to store metrics for each epoch and each client
mean_train_losses = []
mean_train_accuracies = []
mean_train_auc_scores = []

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
logging.info(f"Entropy Selection: {entropy_selection}")
logging.info(f"Client Drop Rate: {drop_rate}")
logging.info(f"Device: {device}")

print(f"Training Process")
# Training loop (FedAvg)
for round in range(global_rounds):
    local_models = []  # Store local models of clients
    logging.info(f"Epoch {round+1}/{global_rounds}")
    selected_clients = select_clients(num_clients, num_clients_to_select, ranked_clients, entropy_selection)

    # Lists to store metrics for each client
    client_train_losses = []
    client_train_accuracies = []
    client_train_auc_scores = []

    if round+1 == round_to_drop_clients:
        # Calculate the number of clients to drop based on the current drop rate
        num_clients_to_drop = int(drop_rate * len(selected_clients))

        # Randomly select clients to drop
        clients_to_drop = np.random.choice(selected_clients, num_clients_to_drop, replace=False)

        # Remove the dropped clients from the selected clients list
        selected_clients = [client_idx for client_idx in selected_clients if client_idx not in clients_to_drop]

        # Log information about dropped clients
        if num_clients_to_drop > 0:
            logging.info(f"{num_clients_to_drop} Clients Disconnected from Training: {clients_to_drop}")

        # Update the round at which you want to drop clients for the next cycle
        round_to_drop_clients += initial_rounds_before_drop
        logging.info(f"{num_clients_to_drop} Clients Out of connection")


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
    #all_client_train_losses.append(client_train_losses)
    #all_client_train_accuracies.append(client_train_accuracies)
    #all_client_train_auc_scores.append(client_train_auc_scores)

    mean_loss = np.mean(client_train_losses)
    mean_train_losses.append(mean_loss)

    mean_accuracy = np.mean(client_train_accuracies)
    mean_train_accuracies.append(mean_accuracy)

    mean_auc_score = np.mean(client_train_auc_scores)
    mean_train_auc_scores.append(mean_auc_score)

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

# Save the model
torch.save(global_model.state_dict(), os.path.join(results_dir, 'global_model.pth'))

# Return the final global model
final_global_model = global_model.to("cpu")

print("Training Complete!")
# Plot mean metrics after the process is completed
plot_mean_metrics(mean_train_losses, mean_train_accuracies, mean_train_auc_scores, results_dir)

# Visualize class distribution in each client
plot_class_distribution(trainset, class_indices, selected_clients, results_dir)
