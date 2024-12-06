#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 21:32:05 2024

@author: nicolasfeitosa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import randint, uniform
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score


##################################################################################
#########                       Funcion Auxiliar                         #########
################################################################################## 

def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))



##################################################################################
#########                             Inputs                             #########
################################################################################## 
print('Grabbing files...')

df_train = pd.read_csv('/Users/nicolasfeitosa/Documents/rossman-sales-prediction/train.csv', low_memory=False)
df_test = pd.read_csv('/Users/nicolasfeitosa/Documents/rossman-sales-prediction/test.csv', low_memory=False)
stores = pd.read_csv('/Users/nicolasfeitosa/Documents/rossman-sales-prediction/store.csv', low_memory=False)
state = pd.read_csv('/Users/nicolasfeitosa/Documents/rossman-sales-prediction/store_states.csv', low_memory=False)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


df_train = pd.merge(df_train, stores, on='Store', how='left')
df_test = pd.merge(df_test, stores, on='Store', how='left')
df_train = pd.merge(df_train, state, on='Store', how='left')
df_test = pd.merge(df_test, state, on='Store', how='left')

# Ordeno mis datos por fecha

df_train = df_train.sort_values(by=['Date'])
df_test = df_test.sort_values(by=['Date'])
df_train = df_train[df_train['Sales']>0]

##################################################################################
#########                           Graficos                             #########
################################################################################## 

# Barras de promo y ventas
ventas_promociones = df_train.groupby('Promo')['Sales'].mean().reset_index()

plt.figure(figsize=(8, 5))
plt.bar(ventas_promociones['Promo'].astype(str), ventas_promociones['Sales'], color=['gray', 'black'])
plt.title('Average Sales on Days With and Without Promotions')
plt.xlabel('Promotion (0 = No Promo, 1 = Promo)')
plt.ylabel('Average Sales')
plt.xticks(rotation=0)
plt.show()

# Dispersión de clientes y ventas
plt.figure(figsize=(10, 6))
plt.scatter(df_train['Customers'], df_train['Sales'], alpha=0.5, color='black')
plt.title('Customers and Sales Dispersion')
plt.xlabel('Customers')
plt.ylabel('Sales')
plt.show()

# Assortment y Store type
ventas_por_tipo = df_train.groupby(['StoreType', 'Assortment'])['Sales'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(data=ventas_por_tipo, x='StoreType', y='Sales', hue='Assortment', palette='dark')

plt.title('Average Sales by Store Type and Assortment')
plt.xlabel('Store Type')
plt.ylabel('Average Sales')
plt.xticks(rotation=0)
plt.legend(title='Assortment')
plt.show()

# Dispersión de competition y sales
df_filt = df_train[df_train['CompetitionDistance'].notna()]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_filt, x='CompetitionDistance', y='Sales', color='black')

plt.title('Relationship Between Competition Distance and Sales')
plt.xlabel('Competition Distance (meters)')
plt.ylabel('Sales')
plt.show()

##################################################################################
#########                           Chequeos                             #########
################################################################################## 

df_train = df_train.drop(columns=['Customers'])

# Chequeamos si los dtypes estan correctos

df_train.dtypes
df_test.dtypes

df_train['Date'] = pd.to_datetime(df_train['Date'])
df_test['Date'] = pd.to_datetime(df_test['Date'])

df_train['PromoInterval'] = df_train['PromoInterval'].astype('str')
df_test['PromoInterval'] = df_test['PromoInterval'].astype('str')

##################################################################################
#########                              NA                                #########
################################################################################## 
print('Checking NaN values...')

nan_counts_train = df_train.isna().sum() # Chequeamos cuantos NA tiene cada columna
nan_counts_test = df_test.isna().sum()

# Vemos que los NA en las columnas Promo2SinceWeek, Promo2SinceYear y PromoInterval son debido a que Promo2 tiene valor 0

promo2_count_train = (df_train['Promo2'] == 0).sum()
promo2_count_test = (df_test['Promo2'] == 0).sum()


test_open = df_test[df_test['Open'].isna()] # Asumimos que la tienda efectivamente estuvo abierta
df_test['Open'] = np.where(df_test['Open'].isna(), 1, df_test['Open'])



##################################################################################
#######                      Ingenieria de atributos                       #######
################################################################################## 
print('Feature Engineering..')

# Sacamos informacion de Date
df_train['Day'] = df_train['Date'].dt.day
df_test['Day'] = df_test['Date'].dt.day

df_train['Month'] = df_train['Date'].dt.month
df_test['Month'] = df_test['Date'].dt.month

df_train['Year'] = df_train['Date'].dt.year
df_test['Year'] = df_test['Date'].dt.year

df_train['DayOfYear'] = df_train['Date'].dt.dayofyear
df_test['DayOfYear'] = df_test['Date'].dt.dayofyear

df_train['Week'] = df_train['Date'].dt.isocalendar().week
df_test['Week'] = df_test['Date'].dt.isocalendar().week

df_train['Is_weekday'] = df_train['Date'].dt.weekday < 5 #0 para el lunes. 6 para el domingo
df_test['Is_weekday'] = df_test['Date'].dt.weekday < 5

df_train['Is_weekday'] = np.where(df_train['Is_weekday'], 1, 0) #0 para el lunes. 6 para el domingo
df_test['Is_weekday'] = np.where(df_test['Is_weekday'], 1, 0)

df_train['Name_of_day'] = df_train['Date'].dt.day_name()
df_test['Name_of_day'] = df_test['Date'].dt.day_name()

# Assortment

df_train['Assortment'] = df_train['Assortment'].map({'a': 1, 'b': 2, 'c': 3})
df_test['Assortment'] = df_test['Assortment'].map({'a': 1, 'b': 2, 'c': 3})

# PromoInterval

df_train['PromoInterval'] = df_train['PromoInterval'].apply(lambda x: x.split(','))
df_test['PromoInterval'] = df_test['PromoInterval'].apply(lambda x: x.split(','))

# Verificar si un mes específico tiene una promoción activa (ejemplo para 'May')

for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
    df_train[f'PromoIn{month}'] = df_train['PromoInterval'].apply(lambda x: month in x)

for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
    df_test[f'PromoIn{month}'] = df_test['PromoInterval'].apply(lambda x: month in x)

# Convierto booleanos a enteros

for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
    df_train[f'PromoIn{month}'] = df_train[f'PromoIn{month}'].astype(int)    

for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
    df_test[f'PromoIn{month}'] = df_test[f'PromoIn{month}'].astype(int)   

print('Sorting NaN values...')
## CompetitionDistance y Promo2Since

df_train['CompetitionDistance'] = df_train['CompetitionDistance'].fillna(-1)
df_test['CompetitionDistance'] = df_test['CompetitionDistance'].fillna(-1)

df_train['CompetitionOpenSinceMonth'] = df_train['CompetitionOpenSinceMonth'].fillna(-1)
df_test['CompetitionOpenSinceMonth'] = df_test['CompetitionOpenSinceMonth'].fillna(-1)

df_train['CompetitionOpenSinceYear'] = df_train['CompetitionOpenSinceYear'].fillna(-1)
df_test['CompetitionOpenSinceYear'] = df_test['CompetitionOpenSinceYear'].fillna(-1)

df_train['Promo2SinceWeek'] = df_train['Promo2SinceWeek'].fillna(-1)
df_test['Promo2SinceWeek'] = df_test['Promo2SinceWeek'].fillna(-1)

df_train['Promo2SinceYear'] = df_train['Promo2SinceYear'].fillna(-1)
df_test['Promo2SinceYear'] = df_test['Promo2SinceYear'].fillna(-1)


# Crear columnas dummy para cada valor único de StateHoliday

df_train['StateHoliday'] = np.where(df_train['StateHoliday']==0, '0', df_train['Open'])
df_test['StateHoliday'] = np.where(df_test['StateHoliday']==0, '0', df_test['Open'])

for value in ['a','b','c','0']:
    df_train['StateHoliday_' + value] = df_train['StateHoliday'].apply(lambda x: 1 if value in x else 0)

for value in ['a','b','c','0']:
    df_test['StateHoliday_' + value] = df_test['StateHoliday'].apply(lambda x: 1 if value in x else 0)
    
# Crear columnas dummy para cada valor único de StoreType
for value in ['a','b','c','d']:
    df_train['StoreType_' + value] = df_train['StoreType'].apply(lambda x: 1 if value in x else 0)

for value in ['a','b','c','d']:
    df_test['StoreType_' + value] = df_test['StoreType'].apply(lambda x: 1 if value in x else 0)

# Crear columnas dummy para cada valor único de Name_of_day
for value in ['Monday','Tuesday','Wednesday','Thursday', 'Friday', 'Saturday', 'Sunday']:
    df_train[value] = df_train['Name_of_day'].apply(lambda x: 1 if value in x else 0)

for value in ['Monday','Tuesday','Wednesday','Thursday', 'Friday', 'Saturday', 'Sunday']:
    df_test[value] = df_test['Name_of_day'].apply(lambda x: 1 if value in x else 0)

# Crear columnas dummy para cada valor único de State
for value in df_train['State'].unique().tolist():
    df_train[value] = df_train['State'].apply(lambda x: 1 if value in x else 0)

for value in df_test['State'].unique().tolist():
    df_test[value] = df_test['State'].apply(lambda x: 1 if value in x else 0)

# Eliminamos columnas que ya no vamos a necesitar (eliminamos algunos estados que no estan presentes en test)

df_train = df_train.drop(columns=['StateHoliday', 'StoreType','Name_of_day', 'PromoInterval', 'Date','State','BE','SN','ST','TH'])
df_test = df_test.drop(columns=['StateHoliday', 'StoreType','Name_of_day', 'PromoInterval', 'Date','State'])

##################################################################################
#########                         Optimizacion                           #########
################################################################################## 
print('Optimizing Model...')


X = df_train.drop(columns=['Sales'])
y = df_train['Sales']


param_dist = {
    "n_estimators": randint(50, 400),  # Número de árboles
    "max_depth": randint(2, 100),  # Profundidad máxima de los árboles
    "min_samples_split": randint(2, 100),  # Mínimo número de muestras para dividir un nodo
    "min_samples_leaf": randint(1, 100),  # Mínimo número de muestras en una hoja
    "max_features": uniform(0.1, 0.9)  # Proporción de características a considerar
}

def random_search(param_dist, n_iter=50):
    exp_results = []
    best_score = np.inf  # Para RMSPE, queremos minimizar
    best_params = None
    best_model = None

    # Usar TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)

    for i in range(n_iter):
        params = {}
        for k, v in param_dist.items():
            if hasattr(v, 'rvs'):  # Verificar si v tiene el método 'rvs'
                params[k] = v.rvs()  # Obtener un valor aleatorio de la distribución
            else:
                params[k] = np.random.choice(v)  # Si no es una distribución, selecciona aleatoriamente
        
        # Asegurarse de que los parámetros sean del tipo correcto
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])

        # Crear el modelo de Random Forest
        model = RandomForestRegressor(**params, random_state=42)

        # Almacenar los RMSPE para cada fold
        rmspe_scores = []

        for train_index, val_index in tscv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            # Ajustar el modelo
            model.fit(X_train, y_train)
            
            # Predecir y calcular RMSPE
            y_val_pred = model.predict(X_val)
            score = rmspe(y_val, y_val_pred)
            rmspe_scores.append(score)

        # Calcular el RMSPE promedio
        avg_rmspe = np.mean(rmspe_scores)

        # Actualizar el mejor modelo y parámetros si es necesario
        if avg_rmspe < best_score:
            best_score = avg_rmspe
            best_params = params
            best_model = model
        print(best_score)
        # Almacenar resultados
        params["avg_rmspe"] = avg_rmspe
        exp_results.append(params)
        print(f"Iteration {i + 1}/{n_iter} - Avg RMSPE: {avg_rmspe:.4f} - Params: {params}")
    
    exp_results = pd.DataFrame(exp_results)
    exp_results = exp_results.sort_values(by="avg_rmspe", ascending=True)
    return best_params, best_score, best_model, exp_results


best_params, best_score, best_model, exp_results = random_search(param_dist, n_iter=100)

print(f"Mejores hiperparámetros: {best_params}")
print(f"Mejor RMSPE: {best_score:.4f}")

##################################################################################
#########                        Modelo Optimo                           #########
################################################################################## 

print('Checking optimized model in train data...')


def fit_transform_models(X, y):
    
    train_size = int(0.8 * len(X))

    # Dividir el DataFrame en 80% train y 20% test
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    models = [
        RandomForestRegressor(n_estimators=300, max_depth=50, max_features=18, n_jobs=-1, random_state=40, min_samples_split=6),
        XGBRegressor(n_estimators=100, max_depth=50, learning_rate=0.1, subsample=0.6, colsample_bytree=0.8, random_state=40, n_jobs=-1)
        ]
    
    for model in models:
        model_name = model.__class__.__name__
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmspe_value = rmspe(y_test, predictions)
        print(f"{model_name} RMSPE: {rmspe_value}")

fit_transform_models(X, y)

##################################################################################
#########                      Prediccion Final                          #########
##################################################################################

print('Forecasting test sales...')

Ids = df_test['Id']
df_test = df_test.drop(columns=['Id'])

new_column_order = ['Store', 'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday', 'Assortment',
                    'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                    'Promo2SinceYear', 'Day', 'Month', 'Year', 'DayOfYear', 'Week',
                    'Is_weekday', 'PromoInJan', 'PromoInFeb', 'PromoInMar', 'PromoInApr',
                    'PromoInMay', 'PromoInJun', 'PromoInJul', 'PromoInAug', 'PromoInSep',
                    'PromoInOct', 'PromoInNov', 'PromoInDec', 'StateHoliday_a',
                    'StateHoliday_b', 'StateHoliday_c', 'StateHoliday_0', 'StoreType_a',
                    'StoreType_b', 'StoreType_c', 'StoreType_d', 'Monday', 'Tuesday',
                    'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'BW', 'NW',
                    'BY', 'SH', 'HE', 'HH', 'RP', 'HB,NI']

# Reordenar las columnas
df_test = df_test[new_column_order]


print('Random Forest')

random_forest = RandomForestRegressor(n_estimators=300, max_depth=50, max_features=18, n_jobs=-1, random_state=40, min_samples_split=6)
random_forest.fit(X,y)
pred = random_forest.predict(df_test)

importances = random_forest.feature_importances_
importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False)
indices = np.argsort(importances)[::-1]

# Grafico importance
plt.figure(figsize=(12, 6))
plt.title('Feature Importance')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.savefig('feat_impo.pdf', format='pdf', bbox_inches='tight')
plt.show()


print('XGBoost')

xgb = XGBRegressor(n_estimators=100, max_depth=50, learning_rate=0.1, subsample=0.6, colsample_bytree=0.8, random_state=40, n_jobs=-1)
xgb.fit(X,y)
pred_xgb = xgb.predict(df_test)


print('Saving predictions...')

# Creamos archivos de envío para Kaggle
submission_rf = pd.DataFrame({'Id': Ids,
                              "Sales": pred})

submission_rf['Id'] = submission_rf['Id'].astype(int)
submission_rf.to_csv("submission_rf.csv", sep=",", index=False)


submission_xgb = pd.DataFrame({'Id': Ids,
                              "Sales": pred_xgb})

submission_xgb['Id'] = submission_xgb['Id'].astype(int)
submission_xgb.to_csv("submission_xgb.csv", sep=",", index=False)

print('Predictions Saved.')

print('Initializing second ML problem..')

##################################################################################
#########                      Segundo Problema                          #########
##################################################################################

# Creamos los percentiles
percentiles = np.percentile(df_train['Sales'], [25, 75])

# Función para categorizar las ventas
def categorize_sales(sales):
    if sales <= percentiles[0]:
        return 'Low'
    elif sales <= percentiles[1]:
        return 'Medium'
    else:
        return 'High'

# Nueva columna con las categorías
df_train['Performance'] = df_train['Sales'].apply(categorize_sales)
df_train.drop(columns=['Sales'], inplace=True)


X_new = df_train.drop(columns=['Performance'])
y_new = df_train['Performance']

train_size = int(0.8 * len(X_new))

# Dividir el DataFrame en 80% train y 20% test
X_train = X_new[:train_size]
y_train = y_new[:train_size]
    
X_test = X_new[train_size:]
y_test = y_new[train_size:]
    
model = CatBoostClassifier(iterations=500, depth=10, learning_rate=0.1, verbose=1, random_seed=50, loss_function='MultiClass')

print('Fitting CatBoost model..')
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)

print('Predicting..')
y_pred = model.predict(X_test)
auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
print(f'AUC (multiclase): {auc:.4f}')

print('Calculating SHAP values...')
# Importancia de características
pool = Pool(X_train, y_train)
shap_values = model.get_feature_importance(pool, type='ShapValues') 

shap_values_contrib = shap_values[:, :, :-1] 

# Calcular contribuciones promediadas por clase
shap_values_avg = np.mean(shap_values_contrib, axis=0)  # Promedio de las contribuciones para cada clase

shap_values_df = pd.DataFrame(shap_values_avg, columns=X_train.columns)

# Agregar la columna de valor esperado si es necesario
expected_value = np.mean(shap_values[:, :, -1], axis=0)
shap_values_df['expected_value'] = expected_value

for i in range(shap_values_avg.shape[0]):  # Iterar sobre las clases
    cla = ['High','Low','Medium']
    plt.figure(figsize=(12, 10))
    
    # Filtrar features donde el promedio de SHAP value es diferente de 0
    mask = shap_values_avg[i] != 0
    filtered_features = X_train.columns[mask]
    filtered_shap_values = shap_values_avg[i][mask]
    
    # Graficar
    plt.barh(filtered_features, filtered_shap_values, color = 'black')
    plt.title('Average Contribution for Class '+ cla[i])
    plt.xlabel('Contributions')
    plt.ylabel('Features')
    plt.axvline(0, color='grey', linewidth=0.8, linestyle='--')
    plt.savefig(cla[i]+'.pdf', format='pdf', bbox_inches='tight')
    plt.show()


print('Run Completed')



