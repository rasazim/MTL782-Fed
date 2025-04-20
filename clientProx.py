import flwr as fl
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss,recall_score,precision_score, f1_score,accuracy_score
import os
import utils
import warnings
import prox_model



class SpamClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_data):
        self.client_id = client_id
        self.X_train, self.y_train = train_data
        self.model = prox_model.CustomLogisticRegression(
            max_iter=1, 
            class_weight='balanced'
        )
        utils.set_initial_params(self.model)


    def get_parameters(self, config):
        return utils.get_model_parameters(self.model)

    def fit(self, parameters, config):
        
        utils.set_model_params(self.model, parameters)
        global_w = np.concatenate((parameters[1], parameters[0].flatten()))
        self.model.prox_mu = config['proximal_mu']
        print(parameters[0].shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train, global_w = global_w)
            print(f"Training finished for round {config['rnd']}")
        return self.get_parameters(config={}), len(self.X_train), {}

    def evaluate(self, parameters, config):
        utils.set_model_params(self.model, parameters)
        loss = log_loss(self.y_train, self.model.predict_proba(self.X_train))
        y_pred = self.model.predict(self.X_train)
        accuracy = accuracy_score(self.y_train,y_pred)
        recall = recall_score(self.y_train,y_pred)
        pres = precision_score(self.y_train,y_pred)
        f1_sc = f1_score(self.y_train,y_pred)
        return loss, len(self.X_train), {"accuracy": accuracy,"recall":recall,"precision":pres,"f1_score":f1_sc}


def load_client_data(client_id):
    client_dir = f"./data_sets/client_{client_id}"
    train_df = pd.read_csv(f"{client_dir}/client_data.csv")
    X_train = train_df.drop("Label", axis=1).values
    y_train = train_df["Label"].values
    return (X_train, y_train)

def client_fn(cid: str):
    train_data = load_client_data(int(cid))
    return SpamClient(cid, train_data)

if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client_fn(os.environ["FLOWER_CLIENT_ID"]),
    )