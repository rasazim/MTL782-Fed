# Federated Learning for Text Classification

This repository contains the code for a Federated Learning project developed for the course MTL782 under the guidance of Prof. Niladri Chatterjee. The project focuses on applying Federated Learning to a text classification task using the SMS Spam Collection dataset.

## Team Members

* Ryan Azim Shaikh
* Keshav Rai
* Kamal Nehra
* Hemant Ramgaria

## Repository Structure

The repository consists of the following main files and data directories:

* `data_sets/`: This directory contains the SMS Spam Collection dataset in two files:
    * `SMS_test`: The test dataset.
    * `SMS_train`: The original training dataset.
    * Client-specific training data will be generated within this directory after running `data_analysis.ipynb`.
* `data_analysis.ipynb`: A Jupyter Notebook responsible for:
    * Loading and cleaning the SMS text data.
    * Removing stopwords and emojis.
    * Vectorizing the text data.
    * Creating a non-IID (Non-Independent and Identically Distributed) split of the training data to simulate the federated setting. The resulting client-specific datasets are stored in appropriately named subfolders within `data_sets/`.
* `client_1.py`: The client-side code for the Flower Federated Learning framework. Each client will run an instance of this script.
* `serverAvg.py`: The server-side code implementing the FedAvg (Federated Averaging) aggregation strategy.
* `serverAdam.py`: The server-side code implementing the FedAdam (Federated Averaging with Adam optimizer) aggregation strategy.
* `serverProx.py`: The server-side code implementing the FedProx (Federated Averaging with Proximal term) aggregation strategy.
* `serverYogi.py`: The server-side code implementing the FedYogi (Federated Averaging with Yogi optimizer) aggregation strategy.
* `utils.py`: Contains various utility functions used across the client and server code.

## Prerequisites

Before running the code, ensure you have the necessary libraries installed. It is highly recommended to create a separate virtual environment to manage the project dependencies.

You can install the required libraries using pip:

```bash
pip install numpy pandas scikit-learn flwr
Running the SimulationThe following steps outline how to run a Federated Learning simulation using one of the implemented server strategies (e.g., FedAvg). Adapt the server script name in Step 2 to run other strategies like FedAdam, FedProx, or FedYogi.Step 1: Open 5 Command Prompt/Terminal WindowsYou will need one terminal for the server and four terminals for the four simulated clients.Step 2: Start the ServerIn the first terminal, navigate to the project directory and run the server script for the desired aggregation strategy. For FedAvg, use the following command:python serverAvg.py
Step 3: Start the ClientsIn the remaining four terminals, navigate to the project directory and run the client script (client_1.py). You need to set the FLOWER_CLIENT_ID environment variable before running each client to assign a unique ID (0 to 3) to each client.Terminal 2 (Client 0):set FLOWER_CLIENT_ID=0
python client_1.py
Terminal 3 (Client 1):set FLOWER_CLIENT_ID=1
python client_1.py
Terminal 4 (Client 2):set FLOWER_CLIENT_ID=2
python client_1.py
Terminal 5 (Client 3):set FLOWER_CLIENT_ID=3
python client_1.py
Once all the server and client scripts are running, the Federated Learning process will begin. The server will coordinate with the clients, and the training progress along with various metrics will be displayed in the server terminal, potentially including plots generated during the process.