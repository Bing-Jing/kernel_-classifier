
import numpy as np
from sklearn.gaussian_process.kernels import RBF,Matern,RationalQuadratic,ExpSineSquared
from scipy.interpolate import interp1d

class funcEnv():
    def __init__(self,funtype = ""):
        self.curFun = None
        self.maxVal = 0
        self.minVal = float('inf')
        self.kernel_lengthscale = 0.1
        self.kernel_var = 1.0
        self.kernel = None
        self.funType = funtype
    def reset(self,sample_point = 2000,upper_bound = 1, lower_bound = 0):
        X = np.linspace(lower_bound-0.1, upper_bound+0.1, num=sample_point)[:, None]
        X1 = np.linspace(lower_bound, upper_bound, num=sample_point)[:, None]
        # 2. Specify the GP kernel (the smoothness of functions)
        # Smaller lengthscale => less smoothness
        # kernel_var = 1.0
        # self.kernel_lengthscale = 0.5 ## modify to 0.1~1.0
        
        if self.funType == "MA":
            self.kernel = self.kernel_var * Matern(self.kernel_lengthscale,nu=1.5)
        elif self.funType == "Exp":
            self.kernel = self.kernel_var * ExpSineSquared(self.kernel_lengthscale,periodicity=0.5)
        elif self.funType == "RQ":
            self.kernel = self.kernel_var * RationalQuadratic(self.kernel_lengthscale,alpha=0.1)
        elif self.funType == "RBF":
            self.kernel = self.kernel_var * RBF(self.kernel_lengthscale)
        else:
            raise ValueError("Unknown fun_type!")
        # print("current function type = {}, length scale = {}".format(self.kernel,self.kernel_lengthscale))
        # 3. Sample true function values for all inputs in X
        trueF = self.sample_true_u_functions(X, self.kernel)
        Y = trueF[0]
        self.curFun = interp1d(X.reshape(-1), Y, kind='cubic')
        self.maxVal = max(self.curFun(X1))
        self.minVal = min(self.curFun(X1))
        return self.curFun
    def getCurFun(self):
        return self.curFun
    # functions sampled from GP
    def sample_true_u_functions(self,X, kernel):
        u_task = np.empty(X.shape[0])
        mu = np.zeros(len(X)) # vector of the means
        C = kernel.__call__(X,X) # covariance matrix
        u_task[:,None] = np.random.multivariate_normal(mu,C).reshape(len(X),1)
        
        return [u_task]
