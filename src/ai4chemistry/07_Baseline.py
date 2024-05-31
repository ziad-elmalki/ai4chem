import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

# Import data
df = pd.read_csv('../../../../docs/data/cycpeptdb_clean.csv', header=0)
filtered_df = df[df['Permeability'] != -10]

# Select the desired columns
columns = ['TPSA',
           'MolWt',
           'NumHAcceptors', 
           'NumHDonors',
           'NumRotatableBonds',
           'MaxPartialCharge',
           'MinPartialCharge',
           'NHOHCount',
           'NOCount',
           'NumHeteroatoms',
           'NumSaturatedCarbocycles',
           'NumSaturatedHeterocycles',
           'NumSaturatedRings',
           'RingCount'] + [col for col in filtered_df.columns if col.startswith('fr_')]

# Create the feature matrix and target vector
X = filtered_df[columns].values
X = np.hstack((X, np.ones((X.shape[0], 1))))  # Add a column of ones for the bias term
y = filtered_df['Permeability'].values

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.7, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

param_grids = {
    'Ridge': {
        'alpha': [0.01, 0.1, 1, 10, 100]
    },
    'Lasso': {
        'alpha': [0.01, 0.1, 1, 10, 100]
    },
    'Random Forest Regressor': {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [None, 10, 20, 30]
    },
    'Gradient Boosting Regressor': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'max_leaf_nodes': [None, 10, 20],
    }
}

models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
}

# Initialize a dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    if name in param_grids:
        print(f"Performing grid search for {name}...")
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_squared_error', error_score='raise')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_  # Retrieve the best fitted model
        print(f"Best parameters for {name}: {grid_search.best_params_}")
    else:
        best_model = model
        best_model.fit(X_train, y_train)  # Fit the model if no hyperparameter tuning is needed
    
    # Predict on validation set
    y_val_pred = best_model.predict(X_val)
    
    # Calculate metrics
    mse = mean_squared_error(y_val, y_val_pred)
    mae = mean_absolute_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)
    
    # Store the results
    results[name] = {"MSE": mse, "MAE": mae, "R2": r2}

# Print the results
for name, metrics in results.items():
    print(f"{name}:")
    print(f"  MSE: {metrics['MSE']}")
    print(f"  MAE: {metrics['MAE']}")
    print(f"  R2: {metrics['R2']}")
    print()

# Evaluate the best model on the test set
# Retrieve the best fitted model from the grid search
best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Test set evaluation (Best Model):")
print(f"  MSE: {test_mse}")
print(f"  MAE: {test_mae}")
print(f"  R2: {test_r2}")

