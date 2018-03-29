import numpy as np
import pandas as pd

class RVM():

    def __init__(self,TaskInfo):

        # parameters
        self.kernel = TaskInfo['Kernel']
        self.kernel_param = np.float(TaskInfo['Kernel_Param'])
        # if TaskInfo.has_key('MaxIts'):
        if 'MaxIts' in TaskInfo:
            self.maxIts = int(TaskInfo['MaxIts'])
        else:
            self.maxIts = 100

        # model info
        self.Weight = None
        self.used_vector = None
        self.alpha = None
        self.beta = None

    def __sum(self,X,dim):
        row,column = X.shape
        r = np.sum(X,dim)
        if dim == 0:
            r = r.reshape((1,column))
        elif dim == 1:
            r = r.reshape((row,1))
        return r

    def __distSqrd(self,X,Y):
        nx = X.shape[0]
        ny = Y.shape[0]
        D2 = np.dot(self.__sum(X ** 2,1),np.ones((1,ny))) + np.dot(np.ones((nx,1)),self.__sum(Y ** 2,1).T) - 2 * np.dot(X,Y.T)
        return D2

    def __sbl_kernelFunction(self,X1,X2,kernel_,lenscal):
        N1 = X1.shape[0]
        N2 = X2.shape[0]
        d = X1.shape[1]

        if kernel_[0] == '+':
            b = np.ones((N1,1))
            kernel_ = kernel_[1:]
        else:
            b = [];

        if (len(kernel_) >= 4) & (kernel_[0:4] == 'poly'):
            p = int(kernel_[4:])
            kernel_ = 'poly'

        if (len(kernel_) >= 5) & (kernel_[0:5] == 'hpoly'):
            p = int(kernel_[5:])
            kernel_ = 'hpoly'

        eta = 1/(lenscal ** 2)

        if kernel_ == 'gauss':
            K = np.exp(-eta * self.__distSqrd(X1,X2))
        elif kernel_ == 'tps':
            r2 = eta * self.__distSqrd(X1,X2)
            K = 0.5 * r2 * np.log(r2 + (r2==0))
        elif kernel_ == 'cauchy':
            r2 = eta * self.__distSqrd(X1,X2)
            K = 1/(1+r2)
        elif kernel_ == 'cubic':
            r2 = eta * self.__distSqrd(X1,X2)
            K = r2 * np.sqrt(r2)
        elif kernel_ == 'r':
            K = np.sqrt(eta) * np.sqrt(self.__distSqrd(X1,X2))
        elif kernel_ =='bubble':
            K = eta * self.__distSqrd(X1,X2)
            K = K<1
        elif kernel_ == 'laplace':
            K = np.exp(-np.sqrt(eta * self.__distSqrd(X1,X2)))
        elif kernel_ == 'poly':
            K = (np.dot(X1,(eta * X2).T)) ** p
        elif kernel_ == 'hpily':
            K = (eta * np.dot(X1,X2.T)) ** p
        elif kernel_ == 'spline':
            K = 1;
            X1 = X1/lenscal
            X2 = X2/lenscal
            for i in range(d):
                XX = np.dot(X1[:,i],X2[:,i])
                Xx1 = np.dot(X1[:,i],np.ones((1,N2)))
                Xx2 = np.dot(np.ones((N1,1)),X2[:,i].T)
                minXX = np.min(Xx1,Xx2)

                K = K * [1 + XX + XX * minXX - (Xx1+Xx2) / 2 * (minXX ** 2) + minXX ** 3 / 3]
        else:
            return(np.NaN)

        return np.column_stack((b,K))

    def __SolveForAlpha(self,s,q,S,Q,old_alpha):
        ALPHA_MAX = 1e12

        n = s.shape[0]
        m = s.shape[1]

        index = range(n)
        C = np.zeros((n,1))
        CC = np.zeros((2*n-1,1))
        for i in range(n):
            t = [d for d in index if d != i ]
            if t == []:
                SS = []
            else:
                SS = -s[t]
            P= np.poly(SS)
            PP = np.convolve(P,P)
            CC = CC + q[i] * q[i] * PP
            C = C + P
        P = np.poly(-s)
        PP = n * np.convolve(P,P).T
        CC0 = np.convolve(P,C[0]).T
        CC = np.column_stack(([0],CC))
        CCC = CC + CC0
        CCC = np.column_stack((CCC,[0]))
        r = np.roots((PP-CCC)[0])

        r_index = [i for i,d in enumerate(np.abs(np.imag(r))) if d == 0]
        r = r[r_index]
        r_index = [i for i,d in enumerate(r) if d > 0]
        r = r[r_index]
        if r.size == 0:
            r = np.column_stack(([ALPHA_MAX]))
        else:
            r = np.column_stack((r,[ALPHA_MAX]))

        r = r[0]

        nn = r.shape[0]

        tt= np.zeros((nn,1))

        L = np.zeros((nn,1))
        for i in range(nn):
            new_alpha = r[i]
            if old_alpha >= ALPHA_MAX and r[i] < ALPHA_MAX:
                tt[i] = 1
                for j in range(n):
                    L[i] = L[i] + np.log(new_alpha/(new_alpha + s[j])) + ((q[j] * q[j]) / (new_alpha + s[j]))
            elif old_alpha < ALPHA_MAX and new_alpha >= ALPHA_MAX:
                tt[i] = 2
                for j in range(n):
                    L[i] = L[i] + (Q[j] * Q[j] / (S[j] - old_alpha)) - np.log(1-(S[j] / old_alpha))
            elif old_alpha < ALPHA_MAX and new_alpha < ALPHA_MAX:
                tt[i] = 3
                for j in range(n):
                    ad = (1/new_alpha) - (1/old_alpha)
                    L[i] = L[i] + (Q[j]*Q[j] / (S[j] + (1/ad))) - np.log(1 + S[j] * ad)
            else:
                tt[i] = 4
                L[i] = 0
        m = np.max(L)
        ind = np.argmax(L)
        new_alpha = r[ind]
        l_inc = L[ind]

        return new_alpha,l_inc

    def __mvrvm(self,PHI,t,maxIts):
        t_row,t_column = t.shape
        PHI_row,PHI_column = PHI.shape

        ALPHA_MAX = 1e12

        # init beta
        epsilon = np.std(t) * 10 / 100
        epsilon = epsilon.reshape((1,t_column))
        beta = 1/(epsilon ** 2)

        N = PHI_row
        M = PHI_column
        P = t_column

        w = np.zeros((M,P))
        PHIt = np.dot(PHI.T,t)

        alpha = np.ones((M,1))
        nonZero = np.ones((M,1),dtype='bool')
        alpha = ALPHA_MAX * alpha

        # choosing the first basis function with non-zero weight
        max_change = -np.inf
        max_k = -1
        max_alpha = -np.inf
        S = np.ones((P,1))
        Q = np.ones((P,1))
        l_inc_vec = np.zeros((M,1))
        alpha_vec = np.zeros((M,1))
        for k in range(M):
            phi = PHI[:,k]
            phi = phi.reshape((PHI_row,1))
            for j in range(P):
                S[j] = (beta[j] * np.dot(phi.T,phi))[0][0]
                tt = t[:,j].reshape((t_row,1))
                Q[j] = (beta[j] * np.dot(phi.T,tt))[0][0]
            if alpha[k] < ALPHA_MAX:
                s = alpha[k] * S / (alpha[k] - S)
                q = alpha[k] * Q / (alpha[k] - S)
            else:
                s = S
                q = Q
            new_alpha,l_inc = self.__SolveForAlpha(s.T,q.T,S.T,Q.T,alpha[k])
            l_inc_vec[k] = l_inc
            alpha_vec[k] = new_alpha

            if max_change < l_inc:
                max_k = k
                max_s = s
                max_q = q
                max_alpha = new_alpha
                max_change = l_inc

        alpha[max_k] = max_alpha

        # start the iterations
        for i in range(maxIts):
            nonZero = (alpha<ALPHA_MAX)
            alpha_nz = alpha[nonZero]
            nz = len(alpha_nz)
            nz_index = [i for i,d in enumerate(nonZero) if d != 0]
            PHI_nz = PHI[:,nz_index]
            SIGMA_nz = np.zeros((nz,nz,P))
            for j in range(P):
                SIGMA_nz[:,:,j] = np.linalg.inv(np.dot(PHI_nz.T,PHI_nz) * beta[j] + np.diag(alpha_nz))

            # choose the basis function which minimises the marginal likelihood
            max_change = -np.inf
            max_k = -1
            max_alpha = -np.inf

            for k in range(M):
                phi = PHI[:,k]
                for j in range(P):
                    S[j] = beta[j] * np.dot(phi.T,phi) - beta[j] * beta[j] * (np.dot(np.dot(np.dot(np.dot(phi.T,PHI_nz),SIGMA_nz[:,:,j]),PHI_nz.T),phi))
                    Q[j] = beta[j] * np.dot(phi.T,t[:,j]) - beta[j] * beta[j] * np.dot(np.dot(np.dot(np.dot(phi.T,PHI_nz),SIGMA_nz[:,:,j]),PHI_nz.T),t[:,j])
                if alpha[k] < ALPHA_MAX:
                    s = alpha[k] * S / (alpha[k] - S)
                    q = alpha[k] * Q / (alpha[k] - S)
                else:
                    s = S
                    q = Q
                new_alpha,l_inc = self.__SolveForAlpha(s.T,q.T,S.T,Q.T,alpha[k])
                l_inc_vec[k] = l_inc
                alpha_vec[k] = new_alpha

                if(max_change < l_inc):
                    max_k = k
                    max_s = s
                    max_q = q
                    max_alpha = new_alpha
                    max_change = l_inc
            alpha[max_k] = max_alpha

            if not nz==0:
                d = np.zeros((1,nz))
                for j in range(P):
                    d = d + np.diag(SIGMA_nz[:,:,j])
                gamma = P - alpha_nz * d
                for j in range(P):
                    gamma = 1- alpha_nz * np.diag(SIGMA_nz[:,:,j])
                    weights = beta[j] * np.dot(np.dot(SIGMA_nz[:,:,j],PHI_nz.T),t[:,j])
                    ED = np.sum((t[:,j] - np.dot(PHI_nz , weights)) ** 2)
                    beta[j] = (N-np.sum(gamma)) / ED

        weights = np.zeros((nz,P))
        for j in range(P):
            weights[:,j] = beta[j] * np.dot(np.dot(SIGMA_nz[:,:,j],PHI_nz.T),t[:,j])
        used = [i for i,d in enumerate(nonZero) if d == True]

        self.Weight = weights
        self.alpha = alpha
        self.beta = beta
        return used

    def fit(self,X,Y):

        if type(X) == pd.DataFrame:
            X = X.values

        # this part because the prediction only for one column so this part will not be use . this will be
        # use in the future, if the prediction should be multi
        if type(Y) == pd.DataFrame:
            Y = Y.values

        if type(Y) == pd.Series:
            Y = Y.values

        if type(Y) == np.ndarray:
            Y = Y.reshape(Y.shape[0],1)


        PHI = self.__sbl_kernelFunction(X, X, self.kernel, self.kernel_param)
        used = self.__mvrvm(PHI, Y, self.maxIts)

        # save the relative vector (this part is from the end of training) because the training part no have the
        # x value (the training part only have the kernel transformed data) so move to here.
        if self.kernel[0] == '+':
            used = [d - 1 for d in used]
            if used[0] != -1:
                self.kernel = self.kernel[1:]
            else:
                used = used[1:]

        self.used_vector = X[used, :]

    def predict(self,X):

        if type(X) == pd.DataFrame:
            X = X.values

        kernel_ = self.kernel
        used_vector = self.used_vector
        weight = self.Weight
        width = self.kernel_param

        PHI = self.__sbl_kernelFunction(X, used_vector, kernel_, width)
        result = np.dot(PHI,weight)

        # return the result, now the prediction is only one target, so it can be sure the result is n*1,
        # so reshape the result (use the max is because the result shape can be n*1 or 1*n whatever)
        # but the max(result.shape) must be n.
        return result.reshape(max(result.shape))

