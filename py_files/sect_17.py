import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from IPython.display import display

# %matplotlib inline
np.random.seed(225)

def regression_formula(x):
    return 5 + 56*x

def make_xy():
    x = np.random.rand(30,1).reshape(30)
    y_randterm = np.random.normal(0,3,30)
    y = 3+ 50* x + y_randterm
    return x,y

def plot_1(x,y,figsize=(10,7)):
    plt.figure(figsize=figsize)
    
    plt.plot(x, y, '.b')
    plt.plot(x, regression_formula(x), '-')
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14);
    plt.show()



    

def errors(x_values, y_values, m, b):
    y_line = (b + m*x_values)
    return (y_values - y_line)

def squared_errors(x_values, y_values, m, b):
    return np.round(errors(x_values, y_values, m, b)**2, 2)

def residual_sum_squares(x_values, y_values, m, b):
    return round(sum(squared_errors(x_values, y_values, m, b)), 2)


def make_table1(x,y,plot=False):

    table = np.zeros((20,2))
    for idx, val in enumerate(range(40, 60)):
        table[idx,0] = val
        table[idx,1] = residual_sum_squares(x, y, val, 1.319)
    if plot:    
        RSS_plot(table)
    return table



def plot_RSS(table,figsize=(10,7)):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.plot(table[:,0], table[:,1], '-')
    plt.xlabel("m-values", fontsize=14)
    plt.ylabel("RSS", fontsize=14)
    plt.title("RSS with changes to slope", fontsize=16);
    plt.show()

def RSS_plot(*args,**kwargs):
    return plot_RSS(*args,**kwargs)

def make_table_sm(x,y):
    table_sm = np.zeros((401,2))
    for idx, val in enumerate(np.linspace(40, 60, 401)):
        table_sm[idx,0] = val
        table_sm[idx,1] = residual_sum_squares(x, y, val, 1.319)
    return table_sm


    
def tan_line(table_sm,start, stop, delta_a,):
    x_dev = np.linspace(start, stop, 100)
    a = (start+stop)/2 
    f_a= table_sm[(table_sm[:,0]==a),1]
    rounded_a_delta_a = round(a+delta_a,2)
    f_a_delta= table_sm[(table_sm[:,0]== (rounded_a_delta_a)),1]
    fprime = (f_a_delta-f_a)/delta_a 
    tan = f_a+fprime*(x_dev-a)
    return fprime, x_dev, tan


def plot_tangent(table_sm,x,y):
    fprime_1, x_dev_1, y_dev_1 = tan_line(table_sm, 41, 43.5, 0.05)
    fprime_2, x_dev_2,  y_dev_2 = tan_line(table_sm, 45, 48, 0.05)
    fprime_3, x_dev_3,  y_dev_3 = tan_line(table_sm, 49, 52, 0.05)


    plt.figure(figsize=(10,7))
    plt.plot(table_sm[:,0], table_sm[:,1], '-')
    plt.plot(x_dev_1, y_dev_1, color = "red",  label = "slope =" + str(fprime_1))
    plt.plot(x_dev_2, y_dev_2, color = "green",  label = "slope =" + str(fprime_2))
    plt.plot(x_dev_3, y_dev_3, color = "orange", label = "slope =" + str(fprime_3))

    plt.xlabel("m-values", fontsize=14)
    plt.ylabel("RSS", fontsize=14)
    plt.legend(loc='upper right', fontsize='large')

    plt.title("RSS with changes to slope", fontsize=16);
    

def sect_17_workflow(display=False):
    if display:
        text ="""
        x,y = make_xy()
        plot_1(x,y)
        table_sm = make_table_sm(x,y)
        RSS_plot(table_sm)
        tangent_plot(table_sm,x,y)
        """
    else:
        x,y = make_xy()
        plot_1(x,y)
        table_sm = make_table_sm(x,y)
        RSS_plot(table_sm)
        plot_tangent(table_sm,x,y)
        
if __name__ =="__main__":
    sect_17_workflow()
    # x,y = make_xy()
    # plot_1(x,y)
    # table_sm = make_table_sm(x,y)
    # RSS_plot(table_sm)
    # tangent_plot(table_sm,x,y)
else:
    print('Run sect_17_workflow(display=True) for lesson workflow. display=False for running workflow.')
    
    
def load_data(
    url = "https://raw.githubusercontent.com/jirvingphd/fsds_100719_cohort_notes/master/datasets/house-prices-advanced-regression-techniques/train.csv"
    ,load_kws=None):
    """Loads train.csv from Kaggle Dataset https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
    """
    import pandas as pd
    if load_kws is None:
        return pd.read_csv(url)
    else:
        return pd.read_csv(url,**load_kws)
    # df.info()
    
 
def plot_reg_xy(x,y,x_label,y_label, figsize=(8,5),
                tick_fmt={"yaxis":"${x:,.0f}"}):
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, s=3, alpha=0.7, label="Raw data")

    ax.set(**{"title":f"{y_label} vs {x_label}",
             "ylabel":y_label,
              "xlabel":x_label})
    
    if tick_fmt is not None:
        if "yaxis" in tick_fmt:
            ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(tick_fmt['yaxis']))
        if "xaxis" in tick_fmt:
            ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(tick_fmt['xaxis']))

    return fig,ax



def plot_reg(df,x_col='GrLivArea',y_col='SalePrice',
             tick_fmt={"yaxis":"${x:,.0f}"}, figsize=(8,5)):
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df[x_col], df[y_col], s=3,
              alpha=0.7, label="raw data")

    ax.set(**{"title":f"{y_col} vs {x_col}",
             "ylabel":y_col,
              "xlabel":x_col})
    
    if tick_fmt is not None:
        if "yaxis" in tick_fmt:
            ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(tick_fmt['yaxis']))
        if "xaxis" in tick_fmt:
            ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(tick_fmt['xaxis']))

    return fig,ax



def regression_formula(x, slope,intercept, return_str=False):
    """
    Returns the predicted y values x using y=mx+b if return_str=False. 
    Returns the string representation of the formula if return_str=True.
    
    Args:
        x (array-like): x data
        slope (scalar): Numeric value for slope
        intercept (scalar): Numeric value for intercept
        return_str (bool, optional): If True, return string formula instead of preds. Defaults to False.
    
    Returns:
        formula: if return_str==True
        y_preds: if return_str==False

    """
    if return_str:
        return f"y= {slope}*x + {intercept}"
    else:
        y_vals = slope*x+intercept
        return y_vals





def loss_function(y,y_hat,kind='MSE'):
    """Get loss from sklearn metrics"""
    import numpy as np
    from sklearn import metrics
    
    if kind in sorted(metrics.SCORERS.keys()):
        loss_func = metrics.get_scorer(kind)
        
    if 'mse' in kind.lower():
        return metrics.mean_squared_error(y,y_hat)
    
    if "mae" in kind:
        return metrics.mean_absolute_error(y,y_hat)
    
    if "rmse" in kind:
        return np.sqrt(metrics.mean_squared_error(y,y_hat))

    


def get_error_for_params(df,x_col='GrLivArea',
                         y_col='SalePrice',
                        slope=0, intercept=0,
                        plot=False,return_xydict=False):
    
    x = df[x_col].values
    y = df[y_col].values
    
    y_pred = regression_formula(x,slope,intercept)
    error = loss_function(y,y_pred)

    if plot:
        plot_preds(x,y,y_pred,label=regression_formula(
            x,slope, intercept, return_str=True))
        
    # else:
    if return_xydict:
        return dict(x=x,y=y,y_pred=y_pred,error=error)
    else:
        return error
    



def plot_preds(x,y,y_pred,label=None):
    
    fig,ax = plot_reg_xy(x,y,"X","Y")

    ax.plot(x,y_pred, ls=":",c='orange',lw=3,
                )
    if label is None:
        title = f"Model Predictions"
        title = f"Model Predictions: {label}"
    ax.set_title(title)
    ax.legend()
    
    
def plot_scatter_vs_regr(df,x_col='GrLivArea',y_col='SalePrice',
                        slope=0, intercept=0):
    
    
    # y_pred = regression_formula(x,slope,intercept)
    res_dict = get_error_for_params(df=df,x_col=x_col,y_col=y_col,
                                 slope=slope,intercept=intercept)
    x = res_dict['x']
    y = res_dict['y']
    error = res_dict['error']
    y_pred = res_dict['y_pred']

    fig,ax = plot_reg(df,x_col=x_col,y_col=y_col)

    x = df[x_col].values
    y= df[y_col].values
    
    
    ax.plot(x,y_pred,
           ls=":",c='orange',lw=3,
            label=regression_formula(x,slope,intercept,return_str=True))
    error = loss_function(y,y_pred)
    ax.set_title(f"Predictions with Error = {error}")
    ax.legend()
#     return fig,ax
    return fig,ax


class Results:
    def __init__(self,result_header=['Slope','Intercept','Loss'] ):
        self._header = result_header
        self._results = [result_header]
        
    def update(self,res_row):
            
        assert len(self._header)==len(res_row)
        self._results.append(res_row)   
        return self.list_to_df()
    
    def list_to_df(self):
        return pd.DataFrame(self._results[1:],columns=self._results[0])
    
    def show(self):
        from IPython.display import display
        display(self.list_to_df().style.set_caption("Results"))
        
        
def test_parameters(x,y,slope,intercept,plot=True):
    y_pred = regression_formula(x=x,slope=slope,intercept=intercept)
    error = loss_function(y,y_pred)
    
    pass
    