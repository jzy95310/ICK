# Reference: Künzel, Sören R., et al. "Metalearners for estimating heterogeneous treatment effects 
# using machine learning." Proceedings of the national academy of sciences 116.10 (2019): 4156-4165.
# Link: https://www.pnas.org/doi/10.1073/pnas.1804597116

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

class X_Learner_RF:

    def __init__(self, classification: bool = False, random_state: int = 0):
        self.M1 = RandomForestClassifier(random_state=random_state) if classification else RandomForestRegressor(random_state=random_state)
        self.M2 = RandomForestClassifier(random_state=random_state) if classification else RandomForestRegressor(random_state=random_state)
        self.M3 = RandomForestClassifier(random_state=random_state) if classification else RandomForestRegressor(random_state=random_state)
        self.M4 = RandomForestClassifier(random_state=random_state) if classification else RandomForestRegressor(random_state=random_state)
        self.g = LogisticRegression(max_iter=2000, random_state=0)
    
    def fit(self, X, T, Y):
        # Step 1: Estimate the response and propensity functions
        X0 = X[(T == 0).ravel()]
        X1 = X[(T == 1).ravel()]
        Y0 = Y[(T == 0).ravel()].ravel()
        Y1 = Y[(T == 1).ravel()].ravel()
        self.M1.fit(X0, Y0)
        self.M2.fit(X1, Y1)
        self.g.fit(X, T.ravel())

        # Step 2: Impute the treatment effect nad estimate the CATE function
        D_hat = np.where((T == 0).ravel(), self.M2.predict(X) - Y.ravel(), Y.ravel() - self.M1.predict(X))
        self.M3.fit(X0, D_hat[(T == 0).ravel()])
        self.M4.fit(X1, D_hat[(T == 1).ravel()])

    def predict(self, X):
        # Return the propensity weighted CATE
        return self.g.predict_proba(X)[:,0] * self.M3.predict(X) + self.g.predict_proba(X)[:,1] * self.M4.predict(X)