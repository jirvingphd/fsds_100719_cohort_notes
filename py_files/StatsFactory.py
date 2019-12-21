class StatsFactory():
    def __init__(self, *args,**kwargs):
        self.help()
        self.data = *args
        self._params = kwargs
        
    def fit(self,verbose=True):
        import statsmodels.api as sm
        from scipy import stats
        import numpy as np
        
        data = self.data
        self._normal = stats.normaltest(data) # normaltest result
        self._equal_var = stats.levene(data)
        self._mean = [np.mean(d) for d in data]
        self._sem = [stats.sem(d) for d in data]
        ## Check Group Sizes
        self._n = [len(x) for x in data]
        
    
    def help(self):
        """Display Hypothesis testing assumption and workflow"""
        workflow="""TO DO: DISPLAY OUTLINE FROM NOTES """
        