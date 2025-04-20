import flwr as fl
from typing import List, Tuple, Dict
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedProx
from flwr.common import (
    EvaluateRes,
    FitRes,
    NDArrays,
    Parameters,
    parameter,
    Scalar,
)
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from typing import Dict
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import os
import utils

# Load the vectorizer
with open("./data_sets/tfidf_vectorizer.pkl", 'rb') as f:
    vectorizer = pickle.load(f)

# Load the test data for server-side evaluation
test_df = pd.read_csv("./data_sets/server/test_data.csv")
X_test = test_df.drop("Label", axis=1).values
y_test = test_df["Label"].values

# Lists to store server-side evaluation metrics
server_round_history = []
server_accuracy_history = []
server_recall_history = []
server_precision_history = []
server_f1_score_history = []
server_log_loss = []

# Lists to store aggregated client-side evaluation metrics
aggregated_round_history = []
aggregated_accuracy_history = []
aggregated_recall_history = []
aggregated_precision_history = []
aggregated_f1_score_history = []
aggregated_log_loss = []


def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Compute weighted average of metrics."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {
        "accuracy": sum(accuracies) / sum(examples),
        "recall": sum(recalls) / sum(examples),
        "precision": sum(precisions) / sum(examples),
        "f1_score": sum(f1_scores) / sum(examples),
    }


def evaluate(
    server_round: int,
    parameters: NDArrays,
    config: Dict[str, Scalar],
) -> Tuple[float, Dict[str, Scalar]]:
    """Evaluate global model on the entire test set."""
    model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced') # Ensure balanced weights for server-side too
    utils.set_model_params(model, parameters)  # Use your utility function to set parameters

    try:
        loss = log_loss(y_test, model.predict_proba(X_test))
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Store server-side metrics
        # server_round_history.append(server_round)
        # server_accuracy_history.append(accuracy)
        # server_recall_history.append(recall)
        # server_precision_history.append(precision)
        # server_f1_score_history.append(f1)

        return loss, {"accuracy": accuracy, "recall": recall, "precision": precision, "f1_score": f1}
    except Exception as e:
        print(f"Error during server-side evaluation: {e}")
        return float('inf'), {"accuracy": 0.0, "recall": 0.0, "precision": 0.0, "f1_score": 0.0}


def fit_round(rnd: int) -> Dict:
    """Send round number to client"""
    return {"rnd": rnd}


def get_on_fit_config_fn(local_epochs: int = 1):
    """Return training configuration dict for each round with specified local epochs."""
    def fit_config(server_round: int):
        """Return training config dict."""
        config = {
            "rnd": server_round,
            "local_epochs": local_epochs,
        }
        return config
    return fit_config


def get_evaluate_fn():
    """Return an evaluation function for server-side evaluation."""
    return evaluate


if __name__ == "__main__":
    # Configure the server with FedProx for FedProx simulation
    print("Running FedProx simulation...")
    # initial_params = fl.common.parameter.ndarrays_to_parameters(utils.get_initial_params())
    strategy_FedProx = FedProx(
        # initial_parameters=initial_params,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=4,
        min_fit_clients=4,
        min_evaluate_clients=4,
        evaluate_metrics_aggregation_fn=weighted_average,
        proximal_mu=0.2,
        on_fit_config_fn=get_on_fit_config_fn(local_epochs=1),
        evaluate_fn=get_evaluate_fn(),
    )

    # Start Flower server and get the history of the training process
    history = fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=strategy_FedProx,
    )

    print("Federated Learning finished.")

    # Extract aggregated client-side metrics from history
    if history and history.losses_distributed and history.metrics_distributed:
        # for round_num, metrics in history.metrics_distributed.items():
        #     print(history.metrics_distributed)
        #     aggregated_round_history.append(round_num)
        #     aggregated_accuracy_history.append(metrics['accuracy'])
        #     aggregated_recall_history.append(metrics['recall'])
        #     aggregated_precision_history.append(metrics['precision'])
        #     aggregated_f1_score_history.append(metrics['f1_score'])
        _,aggregated_log_loss = zip(*history.losses_distributed)
        aggregated_round_history,aggregated_accuracy_history = zip(*history.metrics_distributed['accuracy'])
        _,aggregated_recall_history = zip(*history.metrics_distributed['recall'])
        _,aggregated_precision_history = zip(*history.metrics_distributed['precision'])
        _,aggregated_f1_score_history = zip(*history.metrics_distributed['f1_score'])

    if history and history.losses_centralized and history.metrics_centralized:
        # for round_num, metrics in history.metrics_distributed.items():
        #     print(history.metrics_distributed)
        #     aggregated_round_history.append(round_num)
        #     aggregated_accuracy_history.append(metrics['accuracy'])
        #     aggregated_recall_history.append(metrics['recall'])
        #     aggregated_precision_history.append(metrics['precision'])
        #     aggregated_f1_score_history.append(metrics['f1_score'])
        _,server_log_loss = zip(*history.losses_centralized)
        server_round_history,server_accuracy_history = zip(*history.metrics_centralized['accuracy'])
        _,server_recall_history = zip(*history.metrics_centralized['recall'])
        _,server_precision_history = zip(*history.metrics_centralized['precision'])
        _,server_f1_score_history = zip(*history.metrics_centralized['f1_score'])

    # Plotting
    plt.figure(figsize=(12, 8))

    # print(server_round_history,server_accuracy_history)
    # Plot server-side metrics
    plt.subplot(2, 1, 1)
    plt.plot(server_round_history, server_log_loss, label='Log Loss (Centralized)')
    plt.plot(server_round_history, server_accuracy_history, label='Accuracy (Centralized)')
    plt.plot(server_round_history, server_recall_history, label='Recall (Centralized)')
    plt.plot(server_round_history, server_precision_history, label='Precision (Centralized)')
    plt.plot(server_round_history, server_f1_score_history, label='F1-Score (Centralized)')
    plt.xlabel("Federated Round")
    plt.ylabel("Metric Value")
    plt.title("Server-side Evaluation Metrics")
    plt.legend()
    plt.grid(True)
    # Plot aggregated client-side metrics
    plt.subplot(2, 1, 2)
    plt.plot(aggregated_round_history, aggregated_log_loss, label='Log Loss (Distributed)')
    plt.plot(aggregated_round_history, aggregated_accuracy_history, label='Accuracy (Distributed)')
    plt.plot(aggregated_round_history, aggregated_recall_history, label='Recall (Distributed)')
    plt.plot(aggregated_round_history, aggregated_precision_history, label='Precision (Distributed)')
    plt.plot(aggregated_round_history, aggregated_f1_score_history, label='F1-Score (Distributed)')
    plt.xlabel("Federated Round")
    plt.ylabel("Metric Value")
    plt.title("Aggregated Client-side Evaluation Metrics")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    
    print("-\tlog-loss\taccuracy\trecall\tprecision\tf1")
    print(f"max\t{max(server_log_loss)}\t{max(server_accuracy_history)}\t{max(server_recall_history)}\t{max(server_precision_history)}\t{max(server_f1_score_history)}")
    print(f"min\t{min(server_log_loss)}\t{min(server_accuracy_history)}\t{min(server_recall_history)}\t{min(server_precision_history)}\t{min(server_f1_score_history)}")
