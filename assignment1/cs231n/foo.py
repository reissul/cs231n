import numpy as np

A = np.arange(12).reshape(3,4)
B = np.arange(8).reshape(4,2)

print(A.dot(B))

M, N, D = A.shape[0], A.shape[1], B.shape[1]

r[i, j] is the sum of the ith row in r with the jth column in k

r = A.dot(B)*0
for i in range(M):
    for j in range(N):
        for k in range(D):
            r[i, k] += A[i, j] * B[j, k]
print(r)
