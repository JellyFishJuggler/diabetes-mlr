import numpy as np
import pandas as py

class MultiLinearRegression: 
    
    def __init__(self):
        
        self.theta = None
        
    def fit(self,X,y):
    #Normal Equation: X_transpose, X_t.X.theta = X_t.y
    # theta_0 = bais_column_of_ones
        ip = np.array(X)
        target = np.array(y).reshape(-1,1)
        theta_zero = np.ones((X.shape[0],1))
        ip = np.hstack((theta_zero,ip))
                
        ip_transpose = ip.T
        # ip_inverse = np.linalg.inv(ip)
        # ip_transpose_inverse = np.linalg.inv(ip_inverse)
        
        # X_T_X_inverse = np.linalg.inv(ip_transpose @ ip)
        X_T_X_inverse = np.linalg.pinv(ip_transpose @ ip)
        
        # self.theta = (ip_transpose @ target) @ (X_T_X_inverse)
        self.theta = X_T_X_inverse @ ip_transpose @ target
    
    def predict(self, X):
        
        ip = np.array(X)
        theta_zero = np.ones((ip.shape[0], 1))
        ip = np.hstack((theta_zero,ip))
        theta = self.theta
        
        return ip @ theta
    

df = py.DataFrame({
    "Size(x1)"      :   [4,6,8],
    "Bedroom(x2)"   :   [2,3,1],
    "Price(y)"      :   [19,26,24]
})


model = MultiLinearRegression()

X = df.drop(columns=["Price(y)"])
y = df["Price(y)"]

print(X)
print(y)

X_test = py.DataFrame({
    "Size(x1)": [1600, 2300, 3000],
    "Bedrooms(x2)": [3, 4, 5]
})

model.fit(X,y)
print(model.predict(X_test))        