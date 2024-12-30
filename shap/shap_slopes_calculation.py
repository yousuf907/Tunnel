import os
import argparse
import pandas
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from combine_one_hot import combine_one_hot

import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./experiments_results.csv')
parser.add_argument('--metric', type=str, default='Percentage OOD retained', choices=['Percentage OOD retained', 'Pearson Correlation', 'ID OOD Alignment'])
parser.add_argument('--save_dir', type=str, default='./outputs')

def main(data_path, metric, save_dir):

    print('\n--- Running SHAP calculation for', metric, '---')


    print('\nLoading data from', data_path)
    df = pandas.read_csv(data_path)
    # Define categorical columns
    categorical_columns = ['CNN vs ViT']
    for column in categorical_columns:
        df[column] = df[column].astype('category')
    # Define input and target columnes (X and y)
    x_columns = ['Stem','Spatial Reduction','CNN vs ViT', 'Augmentations', 'Resolution', 'ID Class Count', 'OverParam. Level', 'Depth']
    X = df[x_columns]
    y = df[metric]
    if metric == 'Percentage OOD retained':
        y = y/100
    # One hot encoding for categorical variables
    X_one_hot = pandas.get_dummies(X, sparse=True)
    # Convert non-categorical ones into ordinal
    non_categorical_columns = list(set(x_columns) - set(categorical_columns))
    for column in non_categorical_columns:
        X_one_hot[column] = pandas.factorize(X_one_hot[column], sort=True)[0]
    # Convert all columns to float
    X_one_hot = X_one_hot.astype(float)
    print('\tInput features:', x_columns)
    print('\tTarget metric:', metric)


    print('\nTraining model to predict', metric, 'from the input features')
    model = GradientBoostingRegressor(random_state=0, loss='huber').fit(X_one_hot, y)
    y_pred = model.predict(X_one_hot)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print('\tMSE:', mse)
    print('\tR2:', r2)


    print('\nComputing SHAP values')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_one_hot)
    # Merge categorical SHAP values 
    for column_name in categorical_columns:
        shap_values,sv_occ = combine_one_hot(shap_values, column_name, [column_name in n for n in shap_values.feature_names])


    print('\nCalculating SHAP slopes')
    slopes_dict = {}
    for variable_name in shap_values.feature_names:
        var_index = shap_values.feature_names.index(variable_name)
        var_shap = shap_values.values[:,var_index]
        var_data = shap_values.data[:,var_index]
        var_data = var_data / var_data.max()
        model = LinearRegression().fit(var_data.reshape(-1, 1), var_shap)
        y_pred = model.predict(var_data.reshape(-1, 1))
        slopes_dict[variable_name] = model.coef_[0]
        print('\t', variable_name.ljust(20), 'Slope:', str(model.coef_[0].round(2)).ljust(10), 'R2:', str(r2_score(var_shap, y_pred).round(4)).ljust(10))


    print('\nPlotting SHAP slopes')
    variables = list(slopes_dict.keys())
    slopes = list(slopes_dict.values())
    # Sort variables and slopes by the slopes
    variables, slopes = zip(*sorted(zip(variables, slopes), key=lambda x: x[1]))
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.linewidth"] = 1.5
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    p1=17
    p=15
    bars = ax.barh(variables, slopes, color='#377eb8', alpha=0.75, edgecolor='gray')
    # Add and customizing bar numbers 
    for bar in bars:
        found = False
        if metric == 'Percentage OOD retained':
            limit_a = 0.4
            limit_b = -0.1
            found = True
        if metric == 'Pearson Correlation':
            limit_a = 0.4
            limit_b = -0.1
            found = True
        if metric == 'ID OOD Alignment':
            limit_a = 0.15
            limit_b = -0.05
            found = True
        ha='left'
        if found:
            if bar.get_width() > limit_a:
                ha='right'
            if bar.get_width() < limit_b:
                ha='left'
            if limit_b < bar.get_width()< 0:
                ha='right'
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f' {bar.get_width():.2f} ', va='center', ha=ha,
                fontsize=12, color='k', fontweight='bold')
    ax.set_xlabel('SHAP Slope', fontsize=p1)
    ax.tick_params(axis='x', labelsize=p)
    ax.tick_params(axis='y', labelsize=p1)
    if metric == 'Percentage OOD retained' or metric == 'Pearson Correlation':
        ax.xaxis.set_ticks(np.array([-0.20, 0.00, 0.20, 0.40]))
    elif metric == 'ID OOD Alignment':
        ax.xaxis.set_ticks(np.array([-0.05, 0.00, 0.05, 0.10, 0.15]))
    ax.grid(True, linestyle='-.', linewidth=0.5)
    fig.savefig(save_dir+f'/{metric} shap slopes.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    args = parser.parse_args()
    save_dir = os.path.join(args.save_dir, args.metric)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    main(args.data, args.metric, save_dir)






    
