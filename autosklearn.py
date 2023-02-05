import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import autosklearn.regression
from sklearn.ensemble import RandomForestRegressor

# Load data into a Pandas dataframe
df = pd.read_csv("data.csv")

# Split data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(df.drop("target", axis=1), df["target"], test_size=0.33, random_state=42)

# One-hot encode categorical variables
enc = OneHotEncoder()
train_data = enc.fit_transform(train_data)
test_data = enc.transform(test_data)

# Train the model
automl = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=120, per_run_time_limit=30, tmp_folder='/tmp/autosklearn_regression_example_tmp', output_folder='/tmp/autosklearn_regression_example_out')
automl.fit(train_data, train_labels)

# Get the best model from auto-sklearn
best_model = automl.get_models_with_weights()[0][0]

# Plot the feature importances
if isinstance(best_model, RandomForestRegressor):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)
    plt.barh(range(train_data.shape[1]), importances[indices])
    plt.yticks(range(train_data.shape[1]), [enc.get_feature_names()[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.show()
    
# Plot the average target values for each category of a feature
if isinstance(best_model, RandomForestRegressor):
    feature_index = 0 # Replace with the index of the feature you want to plot
    feature_categories = enc.categories_[feature_index]
    feature_name = enc.get_feature_names()[feature_index]
    avg_target_values = []
    for i, category in enumerate(feature_categories):
        mask = np.array(df[feature_name].astype(str) == category)
        avg_target_values.append(np.mean(df[mask]["target"]))
    plt.bar(feature_categories, avg_target_values)
    plt.xlabel("Category")
    plt.ylabel("Average Target Value")
    plt.show()
