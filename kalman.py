#!/usr/bin/pytho3

from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import time

# Kalman Filter based off of this: https://github.com/zziz/kalman-filter
@jit(nopython=True)
def kf_predict(prev_est, u, F, B, P, Q):
    #new_estimate = np.dot(F, prev_est) + np.dot(B, u)
    new_estimate = np.dot(F, prev_est)
    newP = np.dot(np.dot(F, P), F.T) + Q
    return new_estimate, newP

@jit(nopython=True)
def kf_update(prev_est, meas, F, B, H, Q, R, P, I):
    y = meas - np.dot(H, prev_est)
    S = R + np.dot(H, np.dot(P, H.T))
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    new_est = prev_est + np.dot(K, y)
    newP = np.dot(np.dot(I - np.dot(K, H), P),
                  (I - np.dot(K, H)).T) + np.dot(np.dot(K, R), K.T)

    return new_est, newP

dt = 1.0/60
F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]]).astype(float)
H = np.array([1, 0, 0]).reshape(1, 3).astype(float)
Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]]).astype(float)
R = np.array([0.5]).reshape(1, 1).astype(float)
n = F.shape[1]
P = np.eye(n).astype(float)
B = 0
I = np.eye(n).astype(float)

x = np.linspace(-10, 10, 100)
measurements = - (x**2 + 2*x - 2)  + np.random.normal(0, 2, 100)

predictions = []

est = np.zeros((n, 1))


# This call is just for the kf code to compile with numba
kf_predict(est, 0, F, B, P, Q)
kf_update(est, 0, F, B, H, Q, R, P, I)

for v, k in kf_predict.inspect_llvm().items():
    print(v, k)

for v, k in kf_update.inspect_llvm().items():
    print(v, k)

start = time.time()
for z in measurements:
    est, P = kf_predict(est, 0, F, B, P, Q)
    predictions.append(np.dot(H,  est)[0])
    est, P = kf_update(est, z, F, B, H, Q, R, P, I)

end = time.time()
print("Elapsed Time when running filter = %s" % (end - start))


plt.plot(range(len(measurements)), measurements, label = 'Measurements')
plt.plot(range(len(predictions)), np.array(predictions), label = 'Kalman Filter Prediction')
plt.legend()
plt.show()
