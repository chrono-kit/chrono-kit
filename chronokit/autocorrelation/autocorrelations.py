from chronokit.preprocessing.dataloader import DataLoader
from chronokit.utils.vis_utils import plot_decomp
import numpy as np

class autocorrelations:
    def __init__(self,data):
        self.data = DataLoader(data).to_numpy()

    def p(self,k):
        '''
        Calculates the pearson correlation with lag k (for ACF) with the data passed in the constructor.   
        k (int): lag
        '''
        if k == 0:
            return 1
        n = len(self.data)
        mean = np.mean(self.data)
        p = np.sum((self.data[:n-k]-mean)*(self.data[k:]-mean))/np.sum((self.data-mean)**2)
        return p

    def acf(self, lag):
        '''
        Calculates the autocorrelation function (ACF).
        lag (int): lag
        '''
        acfs = np.array([])
        for i in range(lag+1):
            acfs = np.append(acfs,self.p(i))
        return acfs

    def phi(self,k):
        '''
        Calculates the phi coefficients (for PACF) with the data passed in the constructor using coefficient matrix method.
        k (int): lag
        '''
        if k == 0:
            return [1]
        ac_matrix = np.zeros((k,k))
        sol_vector = np.zeros(k)
        for i in range(k):
            sol_vector[i] = self.p(i+1)
            for j in range(k):
                ac_matrix[i,j] = self.p(np.abs(i-j))
        return np.linalg.solve(ac_matrix,sol_vector)
    
    def pacf(self, lag):
        '''
        Calculates the partial autocorrelation function (PACF).
        lag (int): lag
        '''
        soln = np.array([])
        for i in range(lag+1):
            soln = np.append(soln,self.phi(i)[i-1])
        return soln