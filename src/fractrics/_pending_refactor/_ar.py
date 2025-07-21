from ..helper import OLS
from ..descriptive import acf
import numpy as np

class ar:
    
    def __init__(self, ts, p=0):
        
        # length and lag
        self.ts = np.array(ts)
        self.lag = p
        
        #constructuion of independent and dependent variables
        self.y = ts[p:]
        self.x = np.array([ts[i:i+p] for i in range(len(ts) - p)])
        self.x = np.hstack((self.x, np.ones((self.x.shape[0], 1))))
        self.w = OLS(x=self.x, y=self. y, intercept=False)
        self.pred = np.dot(self.x, self.w)
        self.resid = self.y - self.pred
        self.rsq = 1 - np.sum(self.resid**2)/np.sum((self.y-np.nanmean(self.y))**2)
        self.rsq_adj = 1 - np.sum(self.resid**2*(len(self.y)-1))/np.sum((self.y-np.nanmean(self.y))**2*(len(self.y)-self.lag-1))
        self.acf = acf(self.ts).iloc[:self.lag, :]
        
    def forecast(self, t):
        x_new = self.ts[len(self.ts)-self.lag:]
        y_new = np.empty(t)
        
        for i in range(t):
            x_interc = np.concatenate(([1], x_new))
            y_new[i] = np.dot(x_interc, self.w)
            x_new = np.roll(x_new, shift=-1)
            x_new[-1] = y_new[i]
        return y_new
