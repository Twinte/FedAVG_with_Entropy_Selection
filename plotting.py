# plotting.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_mean_metrics(mean_train_losses, mean_train_accuracies, mean_train_auc_scores, results_dir):
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

def plot_class_distribution(trainset, class_indices, selected_clients, results_dir):
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
