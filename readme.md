# Federated Learning Project – MTL782

This repository contains the codebase for a Federated Learning project developed as part of the course **MTL782** under the guidance of **Prof. Niladri Chatterjee**.

## Team Members

- Ryan Azim Shaikh
- Keshav Rai
- Kamal Nehra
- Hemant Ramgaria


## Project Overview

This project implements federated learning using multiple aggregation algorithms (FedAvg, FedAdam, FedProx, FedYogi) on SMS text data. The workflow includes data preprocessing, non-IID data partitioning, and federated training using the Flower framework.

## Repository Structure

| File/Folder | Description |
| :-- | :-- |
| `data_analysis.ipynb` | Jupyter notebook for data cleaning (removing stopwords, emojis, etc.), vectorization, and non-IID data splitting. Processed data is saved in the `data_sets` directory. |
| `client_1.py` | Client-side code for the Flower federated learning setup. |
| `serverAvg.py` | Server-side code implementing the FedAvg aggregation algorithm. |
| `serverAdam.py` | Server-side code implementing the FedAdam aggregation algorithm. |
| `serverProx.py` | Server-side code implementing the FedProx aggregation algorithm. |
| `serverYogi.py` | Server-side code implementing the FedYogi aggregation algorithm. |
| `utils.py` | Utility functions used across clients and servers. |
| `data_sets/SMS_train` | Training data for SMS classification (partitioned for clients). |
| `data_sets/SMS_test` | Test data for SMS classification. |

## Prerequisites

It is recommended to use a separate Python environment for this project.

Install the required packages:

```bash
pip install numpy pandas scikit-learn flwr
```


## Running the Simulation

Below are the steps to run a federated simulation (e.g., using FedAvg):

1. **Open 5 terminal windows.**
2. **Start the server (Terminal 1):**

```bash
python serverAvg.py
```

3. **Start the clients (Terminals 2–5):**
In each terminal, set the `FLOWER_CLIENT_ID` environment variable and run the client code. Vary the value from 0 to 3 (inclusive):

```bash
set FLOWER_CLIENT_ID=0
python client_1.py
```
Replace `client_1.py` with `clientProx.py` specifically for FedProx.
Repeat in the next terminals with `FLOWER_CLIENT_ID=1`, `FLOWER_CLIENT_ID=2`, and `FLOWER_CLIENT_ID=3`.

4. **Monitor Output:**
The code will execute and display plots of various metrics related to federated learning performance.

## Notes

- Ensure that the data preprocessing step in `data_analysis.ipynb` is completed before running the federated simulation.
- You can switch the server script (`serverAvg.py`, `serverAdam.py`, `serverProx.py`, `serverYogi.py`) to experiment with different aggregation algorithms.
- Run `client_1.py` with `clientProx.py` specifically for FedProx.
---
5. **Experiments:**
-The File BasicExperiments.ipyb run simulation of FedAvg over 100 cleints over IID and Non-IID settings over different paremeters mentioned in Report section 3.3.


For any queries or issues, please contact the Ryan Azim Shaikh (ryanazimsk@gmail.com).

