''' 
Script for training models and selecting the best performing ones 
'''

# basics
import os

# plotting
import matplotlib.pyplot as plt
import numpy as np
# pandas is used to read/process data
import pandas as pd
import rdkit
from drfp import DrfpEncoder
#For confusion Matrix 
import seaborn as sns

# IMPORTS FOR FINGERPRINTS
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import IPythonConsole
from rxnrule.models.utils import load_data, smiles_to_mol, split_reactant, create_fingerprints 
from tqdm import tqdm

# Models
from sklearn.ensemble import RandomForestClassifier

# linear model
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# machine learning dependencies
# train/test split
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

#Save the models 
import pickle 



# WRITE FUNCTIONS

#ADD preprocess function that takes the path of dataframe and returns the train and test sets already splitted 

def preprocess(df_path):
    """Preprocessing function for the dataset."""
    X_data, y_data = load_data(df_path)
    #X_data= X_data_full.sample(2000, random_state=32)
    #y_data = y_data_full.sample(2000, random_state=32)
    split_fp_data, merged_fp_data, Drfp_data = create_fingerprints(X_data)

    # Dictionary to store split data for each fingerprint type
    split_data_dict = {}

    # Split the data we have into training and temporary data (80% train, 20% test) for each fingerprint
    for fingerprint_type, X_FGs in [
        ("Morgan split Fp", split_fp_data),
        ("Morgan Merged Fp", merged_fp_data),
        ("DrFp", Drfp_data),
    ]:
        X_train, X_test, y_train, y_test = train_test_split(
            X_FGs, y_data, test_size=0.2, random_state=420
        )

        # Store split data in the dictionary
        split_data_dict[fingerprint_type] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

    return split_data_dict


def train_models(X_train, y_train, penalty_range, C_range, fit_intercept_range, n_estimators_range,max_depth_range, kernels ,save_path_prefix, results_file_path):
    """Defines and trains a random forest, a Support Vector Machine, and a Logistic Regression model.
    Performs grid search and saves the best model for each type using pickle.

    Args:
        X_train (pd.DataFrame): Contains the Reactant SMILES data we wish to train our models on.
        y_train (pd.DataFrame): Contains the Labels of the Reactants Compatibility (0 for compatible reactants, 1 for incompatible reactants).
        penalty_range (list): List of penalties to search through.
        C_range (list): List of C values to search through.
        fit_intercept_range (list): List of boolean values for fit_intercept.
        save_path_prefix (str): Prefix for the path to save the best models.
        split_data_dict (dict): Dictionary containing split data for each fingerprint type.

    Returns:
        best_models: Dictionary containing the best model for each type.
    """
    rand_for = RandomForestClassifier(random_state=33)
    log_reg = LogisticRegression(max_iter=500)
    svc = SVC(random_state=33)

    models = [rand_for, log_reg, svc]
    best_models = {}

    for model_type, model in zip(['Random Forest', 'Logistic Regression', 'Support Vector'], models):
        best_accuracy = 0.0
        best_model = None
        best_fingerprint_type = None
            
        if model_type=='Logistic Regression':
            param_grid = {
                'penalty': penalty_range,
                'C': C_range,
                'fit_intercept': fit_intercept_range
            }

        if model_type=='Random Forest': 
            param_grid = {
                'n_estimators':n_estimators_range,
                'max_depth' : max_depth_range
            }
        
        if model_type=='Support Vector': 
            param_grid = {
                'C': C_range,
                'kernel' : kernels
        
            }

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        accuracy = grid_search.best_score_

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = grid_search.best_estimator_

        best_models[f'{model_type}'] = best_model
        print(f"Best parameters for {model_type}: {grid_search.best_params_}")

        # Save the best model using pickle
        save_path = f"{save_path_prefix}_{model_type.replace(' ', '_')}.pkl"
        with open(save_path, 'wb') as model_file:
            pickle.dump(best_model, model_file)

        with open(results_file_path, "a") as results_file: 
            # Append results to the text file
            results_file.write(f"Finger Print: {fp}\n")
            results_file.write(f"Model Type: {model_type}\n")
            results_file.write(f"Best Parameters: {grid_search.best_params_}\n")
            results_file.write(f"Best Accuracy: {best_accuracy}\n")
            results_file.write('\n')

    return best_models

def plot_confusion_matrix(y_test, y_pred, classes):
    cm = confusion_matrix(y_pred, y_test) 
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes) 
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("Confusion matrix")
    plt.show()


def evaluate_model(models_dict, X_test, y_test,results_file_path): 
    """Evaluate multiple models and return the best one based on accuracy.

    Args:
        models_dict (dict): Dictionary containing models for different types.
        X_test, y_test (pd.DataFrame): The data to test the models.
        fingerprint_type (str): The type of fingerprint used in training.

    Returns:
        best_model: The best model based on accuracy.
        best_metrics: Dictionary containing metrics of the best model.
    """
    best_model = None
    best_accuracy = 0.0
    best_metrics = {}
    fingerprint = None
    classes = ['Compatible', 'Incompatible']
    for model_name, model in models_dict.items():
        print(f"Evaluating model: {model_name}")
 
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        max_err = max_error(y_test, y_pred)

        print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")
        print(f"MSE: {mse}, Max Error: {max_err}")

        plot_confusion_matrix(y_test, y_pred, classes)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            fingerprint = fp 
            best_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'mse': mse,
                'max_err': max_err
            }

    print(f"The best model is: {type(best_model).__name__}")
    print(f"with accuracy: {best_metrics['accuracy']} and fingerprint {fingerprint}")

    with open(results_file_path, "a") as results_file:
        results_file.write("This is the testing results\n")
        results_file.write(f"The best model is: {type(best_model).__name__}\n") 
        results_file.write(f"with test accuracy: {best_metrics} and fingerprint {fingerprint}\n") 
        results_file.write('\n')

    return best_model, best_metrics


if __name__ == "__main__":
    
    # LOAD DATA
    file_path = "data/processed/generated_data.csv" #/Users/ziadelmalki/Desktop/rxnrule/data/processed/generated_data.csv
    split_data_dict= preprocess(file_path)

    # Set ranges for grid search
    penalty_range = ['l1', 'l2']
    C_range = [0.001, 0.01, 0.1, 1, 10]
    fit_intercept_range = [True, False]
    kernels= ["linear", "poly", "rbf"]
    n_estimators_range = [10,100,200]
    max_depth_range = [3,4,5]
    fingerprint_types = ["Morgan split Fp","Morgan Merged Fp","DrFp"]
    final_models = {}

    for fp in tqdm(fingerprint_types):
        
        print(f"For the FngerPrint: {fp}:")
        X_train = split_data_dict[fp]['X_train']
        y_train = split_data_dict[fp]['y_train']
        results_file_path = 'FullSet_models_training_results.txt'

        #We get the best models based on Grid search, one per classification type 
        best_models = train_models(X_train, y_train , penalty_range, C_range, fit_intercept_range, n_estimators_range, max_depth_range, kernels, fp,results_file_path)
        print(best_models)

        X_test = split_data_dict[fp]['X_test']
        y_test = split_data_dict[fp]['y_test']

        #Out of the three we pick the best per Finger print
        best_model, best_metrics = evaluate_model(best_models,X_test, y_test, results_file_path)
        print(f"The best model is {best_model} with best metrics:{best_metrics}")

        #We store the best one per fingerprint 
        
        final_models[f"{fp}"] = (best_model, best_metrics) 
        print(f'The final three models are: {final_models}')

    with open(results_file_path, "a") as results_file:
        results_file.write(f"The best 3 models are: \n")
        for fingerprint, (model, metrics) in final_models.items():
            results_file.write(f"{model} with {fingerprint} and Results: {metrics}\n") 
        results_file.write('\n')
            
