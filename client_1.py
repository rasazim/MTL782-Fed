import flwr as fl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,recall_score,precision_score, f1_score
import pickle
from sklearn.preprocessing import LabelEncoder
from flwr.common import Metrics
import os
import utils
import warnings

# Load the vectorizer
with open("./data_sets/tfidf_vectorizer.pkl", 'rb') as f:
    vectorizer = pickle.load(f)



# Define Flower client
class SpamClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_data):
        self.client_id = client_id
        self.X_train, self.y_train = train_data
        self.model = LogisticRegression(
            penalty="l2",
            max_iter=1, # local epoch
            warm_start=True, # prevent refreshing weights when fitting
            class_weight='balanced'
        )
        # Setting initial parameters, akin to model.compile for keras models
        utils.set_initial_params(self.model)


    def get_parameters(self, config):
        return utils.get_model_parameters(self.model)

    def fit(self, parameters, config):
        # self.model.coef_, self.model.intercept_ = parameters
        # self.model.fit(self.X_train, self.y_train)
        # return self.get_parameters(config={}), len(self.X_train), {}
        utils.set_model_params(self.model, parameters)
        print(parameters[0].shape)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
            print(f"Training finished for round {config['rnd']}")
        return self.get_parameters(config={}), len(self.X_train), {}

    def evaluate(self, parameters, config):
        # self.model.coef_, self.model.intercept_ = parameters
        # loss = log_loss(self.y_train, self.model.predict_proba(self.X_train))
        # accuracy = self.model.score(self.X_train, self.y_train)
        # return loss, len(self.X_train), {"accuracy": accuracy}
        utils.set_model_params(self.model, parameters)
        loss = log_loss(self.y_train, self.model.predict_proba(self.X_train))
        accuracy = self.model.score(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_train)
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
    # Load data for each client
    train_data = load_client_data(int(cid))
    return SpamClient(cid, train_data)

if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client_fn(os.environ["FLOWER_CLIENT_ID"]),
    )