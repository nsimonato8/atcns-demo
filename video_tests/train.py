import mlflow
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
import ast
from cleaner import DATASET_NUMBER
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Conv3D, BatchNormalization # type: ignore

def load_dataset(n):
    from sklearn.model_selection import train_test_split
    pd_dataset = pd.read_csv(f"video_tests/datasets/dataset_{n}_raw.csv")
    
    X = pd_dataset["X"].apply(lambda x: ast.literal_eval(x.replace("'", "").replace(" ", ", ")))
    X = tf.keras.utils.pad_sequences(sequences=X,
                                     padding='post')
    
    y = np.reshape(pd_dataset["y"].to_numpy(), newshape=(-1, 1))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    
    return (X_train, y_train), (X_test, y_test)

class RNN_HyperModel(kt.HyperModel):

    def build(self, hp):
        units = hp.Int(name="units", min_value=5, max_value=100, step=5)
        learning_rate = hp.Choice(name="learning_rate", values=[1e-3, 1e-4, 5e-4])
        
        model = Sequential()

        model.add(SimpleRNN(units=units))
        
        if units > 20:
            model.add(Dense(units // 5, activation='relu'))
        
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def fit(self, hp, model, *args, **kwargs):
        with mlflow.start_run():
            mlflow.log_params(hp.values)
            mlflow.tensorflow.autolog()
            return model.fit(*args, **kwargs)
        

class LSTM_HyperModel(kt.HyperModel):

    def build(self, hp):
        units = hp.Int(name="units", min_value=5, max_value=100, step=5)
        learning_rate = hp.Choice(name="learning_rate", values=[1e-3, 1e-4, 5e-4])
        
        model = Sequential()

        model.add(LSTM(units=units))
        
        if units > 20:
            model.add(Dense(units // 5, activation='relu'))
        
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def fit(self, hp, model, *args, **kwargs):
        with mlflow.start_run():
            mlflow.log_params(hp.values)
            mlflow.tensorflow.autolog()
            return model.fit(*args, **kwargs)    


class CNN_HyperModel(kt.HyperModel):

    def build(self, hp):
        conv_layers = hp.Int(name="n_conv_layers", min_value=1, max_value=3)
        dense_layers = hp.Int(name="n_dense_layers", min_value=1, max_value=3)
        learning_rate = hp.Choices(name="learning_rate", values=[1e-3, 1e-4, 5e-4])
        
        model = Sequential()

        for n_layer_conv in range(1, conv_layers + 1):
            model.add(Conv3D(filters=hp.Int(name=f"conv_filters_{n_layer_conv}", min_value=3, max_value=10),
                             stride=hp.Int(name=f"conv_stride_{n_layer_conv}", min_value=1, max_value=3),
                             padding="valid"))
            
        for n_layer_dense in range(1, dense_layers + 1):
            model.add(Dense(units=hp.Int(name=f"dense_units_{n_layer_dense}", min_value=1, max_value=3)))
            
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def fit(self, hp, model, *args, **kwargs):
        with mlflow.start_run():
            mlflow.log_params(hp.values)
            mlflow.tensorflow.autolog()
            return model.fit(*args, **kwargs)



if __name__ == '__main__':
    # Initial settings
    mlflow.set_tracking_uri(uri="http://localhost:5000")
    mlflow.set_experiment("Model tuning")

    # Load the dataset
    ((X_train, y_train), (X_test, y_test)) = load_dataset(DATASET_NUMBER)

    # Define the model hyperparameters
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": 42,
    }
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    
    tuner = kt.BayesianOptimization(
        LSTM_HyperModel(),
        max_trials=50,
        overwrite=True,
        objective="val_loss",
        directory="/tmp/tb"
    )

    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])


    best_model = tuner.get_best_models()[0]
    best_hyperparameters= tuner.get_best_hyperparameters()[0].values



    

