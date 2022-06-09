import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_label_occurancies(data, cat_feature):
    cat_value_counts = data[cat_feature].value_counts().sort_values(ascending=True)
    plt.barh(y=cat_value_counts.index, width=cat_value_counts.values)
    plt.title('Histogram of the number of label occurancies')
    plt.show()

def my_log_fun(y):
    return np.log(y+1)

def my_inverse_log_fun(y):
    return np.exp(y) + 1

def my_inverse_log_trans_fun(scaler, y):
    """Inverse function considering both the scaling and the logharithm applied to y before hand. """
    return my_inverse_log_fun(scaler.inverse_transform(y))


def make_feature_importance_df(model, feature_names):
    """ Make a dataframe of the feature importances given the model and the list of features of X. """
    feature_importance = pd.DataFrame(model.feature_importances_, index=feature_names, columns=['Feature Importance'])
    feature_importance = feature_importance.sort_values(by='Feature Importance', ascending=True)
    return feature_importance

def plot_feature_importance(feature_importance_df):
    """Plot feature importance as horizontal barplot given a feature importance dataframe. """
    plt.figure(figsize=(12,12))
    plt.barh(y=feature_importance_df.index, width=feature_importance_df.values.reshape(-1))
    plt.title('Feature Importance')
    plt.show()