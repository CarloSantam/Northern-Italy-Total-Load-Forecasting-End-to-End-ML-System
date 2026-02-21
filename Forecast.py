from italian_holidays import italian_holidays
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from datetime import timedelta
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import os
from datetime import datetime

sys.tracebacklimit = 0  # Disabilita la traceback


import shap

bucket_name='loadforecastingdata'


access_key=os.getenv("access key")

aws_s3_key=os.getenv("secret api key aws")

Merge = pd.read_csv(
    f"s3://{bucket_name}/Data/Final.csv",
    
    storage_options={
       "key": f"{access_key}",
       "secret": f"{aws_s3_key}",
       "client_kwargs": {
           "region_name": "eu-west-1"
       }
       }
    )



Merge['Date']=pd.to_datetime(Merge['Date'])

Merge=Merge.drop_duplicates(subset='Date',keep='first')

hol = italian_holidays()

Merge['Holiday']=Merge['Date'].apply(lambda x:hol.is_holiday(x)).astype('int')

Merge['GiornoSettimana']=Merge['Date'].dt.dayofweek

Merge=Merge.set_index('Date')

Merge['Holiday_prev'] = Merge['Holiday'].shift(freq='1D')

Merge['Holiday_next'] = Merge['Holiday'].shift(-1,freq='1D')

Merge['Bridge']=0


Merge.loc[
    (Merge['GiornoSettimana'] == 0) & 
    (Merge['Holiday_next'] == 1) &
    (Merge['Holiday'] == 0),
    'Holiday'
] = 2

# Case 2: Friday and Thursday is holiday
Merge.loc[
    (Merge['GiornoSettimana'] == 4) &
    (Merge['Holiday_prev'] == 1) &
    (Merge['Holiday'] == 0),
    'Holiday'
] = 2

Merge=Merge.reset_index()

Merge.loc[(Merge['Date'].dt.day==24) & (Merge['Date'].dt.month==12),'Holiday']=1


Merge=Merge.loc[Merge['Date'].dt.year>=2021]

max_d=pd.Timestamp(Merge.loc[Merge['Actual Load'].isna(),'Date'].min())

Merge=Merge.set_index('Date')

# Merge['Actual Load_yearly'] = (
#     Merge['Actual Load']
#     .rolling('365 D')
#     .sum()
# )

# C=Merge.loc[Merge.index<max_d,'Actual Load_yearly'].iloc[-1]

# Merge['Actual Load']*=Merge['Actual Load_yearly']/C

Merge=Merge.reset_index()
Merge=Merge.sort_values(by='Date')
Merge = Merge.set_index('Date')


Merge['Lag_364']=Merge['Actual Load'].shift(freq='364D')

Merge['Lag_365']=Merge['Actual Load'].shift(freq='365D')


Merge['Lag_7']=Merge['Actual Load'].shift(freq='7D')

Merge['Lag_3']=Merge['Actual Load'].shift(freq='3D')

Merge.loc[Merge['Holiday']==1,'Lag_364']=Merge.loc[Merge['Holiday']==1,'Lag_365']

Merge=Merge.drop(columns={'Lag_365'})

Merge=Merge.reset_index()
Merge=Merge.loc[Merge['Date'].dt.year>=2023]

Merge['Mese']=Merge['Date'].dt.month

Merge['Agosto']=Merge['Mese']==8

Merge['Dicembre']=Merge['Mese']==12

Merge['Temp median']=Merge[['temp_k Aosta',
 'temp_k Bologna',
 'temp_k Bolzano',
 'temp_k Genova',
 'temp_k Milano',
 'temp_k Torino',
 'temp_k Trento',
 'temp_k Trieste']].median(axis=1)

Merge['ora']=Merge['Date'].dt.hour


Merge[['ora','GiornoSettimana','Mese','Agosto','Holiday']]=Merge[['ora','GiornoSettimana','Mese','Agosto','Holiday']].astype("category")

for col in ['temp_k Aosta',
 'temp_k Bologna',
 'temp_k Bolzano',
 'temp_k Genova',
 'temp_k Milano',
 'temp_k Torino',
 'temp_k Trento',
 'temp_k Trieste',
 'Temp median']:
    
    Merge[f'HDD_{col[7:]}'] = (18 - Merge[col]).clip(lower=0)
    Merge[f'CDD_{col[7:]}'] = (Merge[col] - 25).clip(lower=0)

Merge_num=Merge.select_dtypes(np.number)

Merge_num['Date']=Merge['Date'].copy()

Merge_num=Merge_num.fillna(0)


# -----------------------------
# 1) Prepare data
# -----------------------------

# # Ensure Date is datetime
Merge = Merge.copy()
Merge["Date"] = pd.to_datetime(Merge["Date"])

# # Define target and features (exclude target and any non-feature columns)
target_col = 'Actual Load'

# ---------------------------------
# 1) Define features and target
# ---------------------------------

# # Drop columns that must not be used as predictors
drop_cols = {"Date", target_col, "Day",'Solar_forecast','Wind Onshore_forecast',
             'Wind Onshore','Solar','Actual Load','Forecasted Load'}  # keep "Day" only if you do NOT want to use it


# ---------------------------------
# 2) Train / Test split by date
# ---------------------------------


train_mask = Merge["Date"] < max_d+timedelta(days=-7)
test_mask  = (Merge["Date"] >= max_d+timedelta(days=-7)) & (Merge["Date"] < max_d)

# train_mask = Merge["Date"] < pd.Timestamp("2025-01-01")
# test_mask  = (Merge["Date"] >= pd.Timestamp("2025-01-01")) & (Merge["Date"] < max_d)

correlation_matrix = ((Merge_num.loc[train_mask].set_index('Date')-Merge_num.loc[train_mask].set_index('Date').mean())/Merge_num.loc[train_mask].set_index('Date').std()).corr(method='spearman')

col_to_keep=list((correlation_matrix['Actual Load'].loc[(np.abs(correlation_matrix['Actual Load'])>0.2)]).index)+['Date']+['ora','GiornoSettimana','Mese','Agosto','Holiday']

Merge=Merge[col_to_keep]

drop_cols = [c for c in drop_cols if c in Merge.columns]

X_raw = Merge.drop(columns=drop_cols)

y = Merge[target_col].astype(float)

X_train_raw = X_raw.loc[train_mask].copy()

y_train2 = y.loc[train_mask].copy()

X_test = X_raw.loc[test_mask].copy()

y_test = y.loc[test_mask].copy()

dates_test = Merge.loc[test_mask, "Date"].copy()

entsoe_fcst=Merge_num.loc[test_mask,'Forecasted Load']

Lag_364=Merge.loc[test_mask,'Lag_364']

# ---------------------------------
# 3) Handle missing values
# ---------------------------------

# Fill NaNs (consistent with your original logic)
X_train2 = X_train_raw
X_test  = X_test

q=0.5 

xgb = XGBRegressor(
    objective="reg:quantileerror",
    quantile_alpha=q,
    random_state=42,
    learning_rate=0.05,
    n_estimators=100,
    max_depth=3,
    n_jobs=-1,
    enable_categorical=True

)

X_train2[['Mese','Agosto','Holiday']]=X_train2[['Mese','Agosto','Holiday']].astype(int).astype("category")

X_test[['Mese','Agosto','Holiday']]=X_test[['Mese','Agosto','Holiday']].astype(int).astype("category")

X_test[X_test.select_dtypes('category').columns]=X_test[X_test.select_dtypes('category').columns].astype(int).astype("category")

X_train2[X_test.select_dtypes('category').columns]=X_train2[X_train2.select_dtypes('category').columns].astype(int).astype("category")

xgb.fit(
    X_train2, (y_train2),
    verbose=False)

# -----------------------------
# 5) Predict 2025
# -----------------------------

y_pred_2025_xgboost = (xgb.predict(X_test))

cat_model = CatBoostRegressor(
    loss_function="RMSE",
    random_seed=42,
    learning_rate=0.1,
    iterations=100,
    depth=3,
    verbose=False,
    cat_features=['ora','GiornoSettimana','Mese','Agosto','Holiday']
)

cat_model.fit(X_train2, (y_train2).interpolate(method='linear'))

y_pred_2025_catboost = (cat_model.predict(X_test))

rf_model = RandomForestRegressor(
    n_estimators=100,        # number of trees
    max_depth=12,          # let trees grow (or set 8–15)
    max_features="sqrt",     # default and usually good
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train2, (y_train2).interpolate(method='linear'))

# Build table
fi = pd.Series(rf_model.feature_importances_, index=X_train2.columns)

# Sort descending
fi = fi.sort_values(ascending=True)

# # Plot
plt.figure()
fi.plot(kind="barh")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()



y_pred_2025_rf = np.quantile(np.array([tree.predict(X_test) for tree in rf_model.estimators_]),0.5,axis=0)

pred_df = pd.DataFrame({
    "Date": dates_test.values,
    "y_true": y_test.values,
    "Xgb": y_pred_2025_xgboost,
    "Cat": y_pred_2025_catboost,
    "RF": y_pred_2025_rf,
    "Entsoe": entsoe_fcst,
    "Lag_364":Lag_364

}).sort_values("Date")

pred_df = pred_df.replace([np.inf, -np.inf], np.nan).dropna()

pred_df['mese']=pred_df['Date'].dt.month

pred_df['Anno']=pred_df['Date'].dt.year


pred_df['tot']=pred_df.groupby(['Anno','mese'])['y_true'].transform(sum)

pred_df['tot']=pred_df['y_true'].sum()

pred_df['mean']=(2/3*pred_df['Xgb']+1/3*pred_df['RF'])

pred_df['mix']=pred_df['Xgb']

pred_df.loc[pred_df['Date'].dt.hour.isin(list(range(10,14))),'mix']=pred_df.loc[pred_df['Date'].dt.hour.isin(list(range(10,14))),'RF']

def wape(pred_df):
    for method in ['Xgb','Cat','RF','mean','mix','Entsoe','Lag_364']:
        pred_df[f'MAE_{method}']=np.abs(pred_df[f'{method}']-pred_df['y_true'])/pred_df['tot']        
        pred_df[f'WAPE_{method}']=pred_df[f'MAE_{method}'].sum()
        
    return pred_df[['WAPE_Xgb','WAPE_Cat','WAPE_RF','WAPE_mean','WAPE_mix','WAPE_Entsoe','WAPE_Lag_364']].drop_duplicates()

WAPE=wape(pred_df)

pred_df_=pred_df.melt(id_vars=['Date'],value_vars=['Xgb','Cat','RF','mean','mix','Entsoe','y_true','Lag_364'],
                         var_name='Model',value_name='Forecast')


px.defaults.template = "simple_white"

fig=px.line(pred_df_,x='Date',y='Forecast',color='Model')

fig.show(renderer='browser')

subset = WAPE.iloc[:, 1:3]

min_column_per_row = subset.idxmin(axis=1).iloc[0]

test_mask  = (Merge["Date"] > max_d) & (Merge["Date"] < max_d+timedelta(days=3))

X_test = X_raw.loc[test_mask].copy()

y_forecasted=np.ones(len(X_test))

model=''

if min_column_per_row=='WAPE_Xgb':
    model='Xgb'
    best_model=xgb
    X_test[X_test.select_dtypes('category').columns]=X_test[X_test.select_dtypes('category').columns].astype(int).astype("category")

    y_forecasted=best_model.predict(X_test)
    
elif min_column_per_row=='WAPE_Cat':
    model='Cat'
    best_model=cat_model
    y_forecasted=best_model.predict(X_test)
    
elif min_column_per_row=='WAPE_RF':
    model='RF'
    best_model=rf_model
    y_forecasted=np.quantile([tree.predict(X_test) for tree in best_model.estimators_],0.5,axis=0)

dates_test = Merge.loc[test_mask, "Date"].copy()

entsoe = Merge.loc[test_mask, 'Forecasted Load'].copy()


pred_df_end = pd.DataFrame({
    "Date": dates_test.values,
    "Forecasted":y_forecasted,
    "Entsoe":entsoe

}).sort_values("Date")


# Create TreeExplainer for Random Forest
explainer = shap.TreeExplainer(best_model,feature_perturbation='tree_path_dependent')

# Compute SHAP values
shap_values = explainer.shap_values(X_test)

Shap=pd.DataFrame(shap_values)

Shap.columns=X_test.columns

Shap['Date']=dates_test.values

End=pred_df_end.merge(Shap,on=['Date'],how='left')

End.columns=End.columns+'_shap'

End=End.rename(columns={'Date_shap':'Date','Forecasted_shap':'Forecasted'})

End=End.reset_index().drop(columns={'index'})

End.drop(columns={'Entsoe_shap'}).to_csv(
    f"s3://{bucket_name}/Data/End.csv",
    index=False,  # # Avoid writing index column
    storage_options={
        "key": access_key,
        "secret": aws_s3_key,
        "client_kwargs": {
            "region_name": "eu-west-1"
        }
    }
)

px.defaults.template = "simple_white"


pred_df_end_=pred_df_end.melt(id_vars=['Date'],value_vars=['Forecasted','Entsoe'],
                         var_name='Model',value_name='Forecast')

fig=px.line(pred_df_end_,x='Date',y='Forecast',color='Model')

fig.show(renderer='browser')

C1=Merge.loc[train_mask,['Date','Actual Load']].rename(columns={'Actual Load':'Forecasted'})

C1['status']='Train'

C2=pred_df_.loc[pred_df_['Model']==model,['Date','Forecast']].rename(columns={'Forecast':'Forecasted'})

C2['status']='Val'

C3=End[['Date', 'Forecasted']]

C3['status']='Forecast'

Prev_completa=pd.concat([C1,C2,C3])

Prev_completa.to_csv(
    f"s3://{bucket_name}/Data/Prev_completa.csv",
    index=False,  # # Avoid writing index column
    storage_options={
        "key": access_key,
        "secret": aws_s3_key,
        "client_kwargs": {
            "region_name": "eu-west-1"
        }
    }
)