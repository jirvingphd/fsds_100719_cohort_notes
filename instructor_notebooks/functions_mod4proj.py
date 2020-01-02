import fsds_100719 as fs
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from IPython.display import display 


def melt_data(df): #formerly called it melt_data_student with REED
    melted = pd.melt(df, id_vars=['RegionID','RegionName', 'City', 'State', 'Metro', 'CountyName', 
                                  'SizeRank'], var_name='Month', value_name='MeanValue')
    melted['Month'] = pd.to_datetime(melted['Month'], format='%Y-%m')
    melted = melted.dropna(subset=['MeanValue'])
    return melted


def make_dateindex(df_to_add_index, index_col='Month',index_name = 'date',
                   drop=True,freq=None,verbose=True):
    
    df = df_to_add_index.copy()
    df.reset_index(drop=True)
    
    ## Make datetime column (to make into index)
    df[index_name] = pd.to_datetime(df[index_col], errors='coerce')
    
#     if index_name != df.index.name:
    df = df.set_index(index_name,drop=drop)
    
#     if freq is not None:
#         try:
#             df = df.resample(freq,fill_method='ffill').mean()
#         except Exception as e:
#             print(f"Error: {e}, index.name={df.index.name}")
        
    if verbose:
        display(df.index)
        
    return df


def get_train_test_split_index(ts, TEST_SIZE=0.2):
    import math
    idx_split = math.floor(len(ts.index)*(1-TEST_SIZE))
    return idx_split


def thiels_U(ys_true=None, ys_pred=None,display_equation=True,display_table=True):
    """Calculate's Thiel's U metric for forecasting accuracy.
    Accepts true values and predicted values.
    Returns Thiel's U"""


    from IPython.display import Markdown, Latex, display
    import numpy as np
    display(Markdown(""))
    eqn=" $$U = \\sqrt{\\frac{ \\sum_{t=1 }^{n-1}\\left(\\frac{\\bar{Y}_{t+1} - Y_{t+1}}{Y_t}\\right)^2}{\\sum_{t=1 }^{n-1}\\left(\\frac{Y_{t+1} - Y_{t}}{Y_t}\\right)^2}}$$"

    # url="['Explanation'](https://docs.oracle.com/cd/E57185_01/CBREG/ch06s02s03s04.html)"
    markdown_explanation ="|Thiel's U Value | Interpretation |\n\
    | --- | --- |\n\
    | <1 | Forecasting is better than guessing| \n\
    | 1 | Forecasting is about as good as guessing| \n\
    |>1 | Forecasting is worse than guessing| \n"


    if display_equation and display_table:
        display(Latex(eqn),Markdown(markdown_explanation))#, Latex(eqn))
    elif display_equation:
        display(Latex(eqn))
    elif display_table:
        display(Markdown(markdown_explanation))

    if ys_true is None and ys_pred is None:
        return

    # sum_list = []
    num_list=[]
    denom_list=[]
    for t in range(len(ys_true)-1):
        num_exp = (ys_pred[t+1] - ys_true[t+1])/ys_true[t]
        num_list.append([num_exp**2])
        denom_exp = (ys_true[t+1] - ys_true[t])/ys_true[t]
        denom_list.append([denom_exp**2])
    U = np.sqrt( np.sum(num_list) / np.sum(denom_list))
    return U


def get_model_metrics(true,preds,train,explain_U=False):
    from sklearn.metrics import r2_score, mean_squared_error
    
    results = [['Metric','Value']]
    r2 = r2_score(true,preds)
    results.append(['R^2',r2])
    
    rmse = np.sqrt(mean_squared_error(true,preds))
    results.append(['RMSE',rmse])
    
    if explain_U:
         U_kws = dict(display_equation=True, display_table=True)
    else:
         U_kws = dict(display_equation=False, display_table=False)
    U = thiels_U(true,preds,**U_kws)
    results.append(["Thiel's U",U])
    
    
    fig,axes=plt.subplots(ncols=2,figsize=(12,6))

#     axes[0] = plt.subplot2grid(shape=(1,4),loc=(0,0),colspan=3, fig=fig)
#     axes[1] = plt.subplot2grid(shape=(1,4),loc=(0,3),colspan=1, fig=fig)

    ax=axes[0]
    ax.plot(true,label='Test Data')
    ax.plot(preds, label='Model Forecast')
    ax.plot(train,label='Training Data')

    ax.legend()
    
    ax=axes[1]
    ax.scatter(x=true.index,y=true.values,label='Test Data')
    ax.scatter(x=preds.index,y=preds.values,label='Training Data')

    #     ax.axis('off')
    
    res = fs.list2df(results)#,index_col='Metric')
    
#     pd.plotting.table(ax,res,**{'fontsize':40})#,loc='right')
    plt.tight_layout()
    
    return res
    
    
    
    
## Lab Function
def stationarity_check(TS,plot=True,col=None):
    """From: https://learn.co/tracks/data-science-career-v2/module-4-a-complete-data-science-project-using-multiple-regression/working-with-time-series-data/time-series-decomposition
    """
    
    # Import adfuller
    from statsmodels.tsa.stattools import adfuller

    if col is not None:
        # Perform the Dickey Fuller Test
        dftest = adfuller(TS[col]) # change the passengers column as required 
    else:
        dftest=adfuller(TS)
 
    if plot:
        # Calculate rolling statistics
        rolmean = TS.rolling(window = 8, center = False).mean()
        rolstd = TS.rolling(window = 8, center = False).std()

        #Plot rolling statistics:
        fig = plt.figure(figsize=(12,6))
        orig = plt.plot(TS, color='blue',label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
#     plt.show(block=False)
    
    # Print Dickey-Fuller test results
    print ('Results of Dickey-Fuller Test:')

    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
        

    sig = dfoutput['p-value']<.05
    print (dfoutput)
    print()
    if sig:
        print(f"[i] p-val {dfoutput['p-value'].round(4)} is <.05, so we reject the null hypothesis.")
        print("\tThe time series is NOT stationary.")
    else:
        print(f"[i] p-val {dfoutput['p-value'].round(4)} is >.05, therefore we support the null hypothesis.")
        print('\tThe time series IS stationary.')
    
    
    
    return dfoutput



def plotly_timeseries(df,x='datetime',y='MeanValue',color='RegionID',
                  line_group='State'):
    from plotly import express as px
    import plotly.graph_objects as go

    pfig = px.line(data_frame=df,x=x,y=y,color=color,
                  line_group=line_group)

    pfig.update_layout(title_text='Time Series with Rangeslider',
                      xaxis_rangeslider_visible=True)

    # Add range slider
    pfig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    return pfig




def meta_grid_search(ts, TEST_SIZE=0.2,model_kws={},verbose=True,return_kws=False):
    import pmdarima as pm
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    ## Train Test Split
    idx_split = get_train_test_split_index(ts,TEST_SIZE=TEST_SIZE)

    ts_train = ts.iloc[:idx_split].copy()
    ts_test = ts.iloc[idx_split:].copy()
    
    
    ## Combine Default kwargs and model_kws
    model_kwargs = dict(start_p=0,start_q=0,start_P=0,start_Q=0, 
                          max_p=5 ,max_q =6 ,max_P= 5,max_Q=5,max_D=3,
                          suppress_warnings=True, stepwise=False, trace=False,
                          m=6,seasonal=True, with_intercept=True,
                          stionarity=False)
    
    for k,v in model_kws.items():
        model_kwargs[k]=v
    
    if verbose:
        print("pm.auto_arima args:")
        print(model_kwargs)

    model = pm.auto_arima(ts_train,**model_kwargs)
    display(model.summary())
    
    
    model_sarimax = SARIMAX(ts_train,
                            **model.get_params()).fit()
    
    preds = model_sarimax.predict(ts_test.index[0],ts_test.index[-1])
    res = get_model_metrics(ts_test,preds,ts_train)
    display(res)
    
    return model_sarimax
    
#     model_sarimax.summary()
