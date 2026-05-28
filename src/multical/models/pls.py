import numpy as np
from ..utils import zscore_matlab_style

class PLS:
    def __init__(self):
        pass

    def nipals(self, X, Y, h):
        """
        Implements the NIPALS algorithm from pls.sci.
        Strict 1:1 translation of the mathematical operations.
        """
        m, n = X.shape
        # Ensure Y is 2D
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        E = X.copy()
        F = Y.copy()
        
        ssX = np.sum(E**2)
        ssY = np.sum(F**2)
        
        # Initialize output matrices
        W = np.zeros((n, h))
        P = np.zeros((n, h))
        T = np.zeros((m, h))
        U = np.zeros((Y.shape[1], h)) # Placeholder for compatibility, shape might vary if Y has >1 col
        
        U_scores = np.zeros((m, h)) 
        Q_loadings = np.zeros((Y.shape[1], h)) 

        r2X = np.zeros(h)
        r2Y = np.zeros(h)

        for i in range(h):
            u = F[:, 0].reshape(-1, 1)
            uold = np.ones((m, 1)) * 100
            
            # Convergence loop
            while np.linalg.norm(uold - u) > 1e-5:
                uold = u.copy()
                w = E.T @ u
                w = w / np.linalg.norm(w)
                t = E @ w
                q = F.T @ t / (t.T @ t)
                u = F @ q / np.sqrt(q.T @ q)
            
            p = E.T @ t / (t.T @ t)
            
            # Store components
            W[:, i:i+1] = w
            P[:, i:i+1] = p
            T[:, i:i+1] = t
            U_scores[:, i:i+1] = u
            Q_loadings[:, i:i+1] = q
            
            # Deflation
            E = E - t @ p.T
            F = F - t @ q.T
            
            # Variance explained (Strict Scilab logic)
            # r2X(i) = 100*(t'*t*(p'*p)/ssX)
            r2X[i] = 100 * ((t.T @ t) * (p.T @ p) / ssX).item()
            r2Y[i] = 100 * ((t.T @ t) * (q.T @ q) / ssY).item()

        # Regression Coefficients Calculation: B = W * pinv(P' * W) * Q'
        # Scilab: B = W*pinv(P'*W)*Q';
        B = W @ np.linalg.pinv(P.T @ W) @ Q_loadings.T
        
        return B, T, P, U_scores, Q_loadings, W, r2X, r2Y

    def compute_outlier_metrics(self, X, T, P, conf_level=0.95):
        """
        Computes Hotelling's T2 and Q-residuals (SPE) and their confidence limits.
        """
        from scipy.stats import f, chi2
        
        n_samples, n_components = T.shape
        
        # 1. Hotelling's T2
        # T2 = sum( (t_i / std(t))^2 )
        score_var = np.var(T, axis=0, ddof=1)
        score_var[score_var == 0] = 1e-10
        T2 = np.sum((T ** 2) / score_var, axis=1)
        
        # T2 Limit (using F distribution)
        if n_samples > n_components:
            t2_limit = (n_components * (n_samples - 1) / (n_samples - n_components)) * f.ppf(conf_level, n_components, n_samples - n_components)
        else:
            t2_limit = np.inf
            
        # 2. Q-residuals (SPE - Squared Prediction Error)
        X_recon = T @ P.T
        E = X - X_recon
        Q = np.sum(E ** 2, axis=1)
        
        # Q Limit (Box empirical approximation based on chi-squared)
        m = np.mean(Q)
        v = np.var(Q, ddof=1)
        if v == 0:
            v = 1e-10
            
        g = v / (2 * m)
        h_dof = (2 * m ** 2) / v
        q_limit = g * chi2.ppf(conf_level, h_dof)
        
        return T2, t2_limit, Q, q_limit

    def predict_model(self, X, Y, k, Xt=None, teste_switch=1):
        """
        Equivalent to pls_model.sci
        teste_switch: 0 or 1 (normalization mode)
        """
        # Ensure Inputs are 2D
        X = np.array(X)
        Y = np.array(Y)
        if Y.ndim == 1: Y = Y.reshape(-1, 1)
        if Xt is not None: Xt = np.array(Xt)
        
        if teste_switch == 0:
            # Case 0: Normalize X independent of Xt
            Xnorm, Xmed, Xsig = zscore_matlab_style(X)
            Ynorm, Ymed, Ysig = zscore_matlab_style(Y)
            
            B, T, _, _, _, _, _, _ = self.nipals(Xnorm, Ynorm, k)
            
            Ynormp = Xnorm @ B
            # Denormalize Yp
            Yp = Ynormp * Ysig + Ymed
            
            Ytp = None
            if Xt is not None:
                # Xt normalized using X's parameters
                Xtnorm = (Xt - Xmed) / Xsig
                Ytnormp = Xtnorm @ B
                Ytp = Ytnormp * Ysig + Ymed
                
            return Yp, Ytp, {'Xmed': Xmed, 'Xsig': Xsig, 'Ymed': Ymed, 'Ysig': Ysig, 'Beta': B, 'T': T}

        elif teste_switch == 1:
            # Case 1: Normalize [X; Xt] together
            n = X.shape[0]
            if Xt is not None:
                Combined = np.vstack([X, Xt])
                Combined_norm, Xmed, Xsig = zscore_matlab_style(Combined)
                Xnorm = Combined_norm[:n, :]
                Xtnorm = Combined_norm[n:, :]
            else:
                # Fallback if no Xt provided but switch is 1
                Xnorm, Xmed, Xsig = zscore_matlab_style(X)
                Xtnorm = None
                
            Ynorm, Ymed, Ysig = zscore_matlab_style(Y)
            
            B, T, _, _, _, _, _, _ = self.nipals(Xnorm, Ynorm, k)
            
            Ynormp = Xnorm @ B
            Yp = Ynormp * Ysig + Ymed
            
            Ytp = None
            if Xt is not None and Xtnorm is not None:
                Ytnormp = Xtnorm @ B
                Ytp = Ytnormp * Ysig + Ymed
                
            return Yp, Ytp, {'Xmed': Xmed, 'Xsig': Xsig, 'Ymed': Ymed, 'Ysig': Ysig, 'Beta': B, 'T': T}

    def compute_outlier_metrics(self, X, T, P, conf_level=0.95):
        """
        Calculates Hotelling's T^2 and Q-residuals (SPE) for outlier detection.
        Returns:
            T2: Hotelling's T^2 for each sample.
            t2_limit: T^2 threshold for given confidence level.
            Q_res: Q-residual for each sample.
            q_limit: Q-residual threshold for given confidence level.
        """
        import scipy.stats as stats
        n_samples, n_features = X.shape
        n_components = T.shape[1]

        # Hotelling's T^2
        # T2 = sum( (t_ia / s_a)^2 ) for a=1..A
        # Alternatively, T = X * W*, T2 = T * inv(cov(T)) * T' (for z-scored T, variance of T columns is eigenvalues of cross product)
        
        # Calculate variance of scores (eigenvalues)
        T_var = np.var(T, axis=0, ddof=1)
        # Handle zero variance
        T_var[T_var == 0] = np.finfo(float).eps
        T2 = np.sum((T ** 2) / T_var, axis=1)

        # Hotelling limit (F-distribution)
        F_val = stats.f.ppf(conf_level, n_components, n_samples - n_components)
        t2_limit = n_components * (n_samples - 1) / (n_samples - n_components) * F_val

        # Q-residuals (Squared Prediction Error)
        # E = X - T * P'
        E = X - np.dot(T, P.T)
        Q_res = np.sum(E ** 2, axis=1)

        # Q-residual limit using Box's approximation or Jackson-Mudholkar
        # Using simple Chi-squared approximation: 
        # Mean and Variance of Q
        m = np.mean(Q_res)
        v = np.var(Q_res, ddof=1)
        if v == 0: v = np.finfo(float).eps
        # Degrees of freedom for chi-square
        h0 = 2 * (m ** 2) / v
        # Scale for chi-square
        scale = v / (2 * m)
        q_limit = scale * stats.chi2.ppf(conf_level, h0)
        
        return T2, t2_limit, Q_res, q_limit
