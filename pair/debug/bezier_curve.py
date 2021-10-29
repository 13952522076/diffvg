"""thiss file is for ploting bezier curve with for points"""
import torch
import matplotlib.pyplot as plt


def bezier_point(A,B,C,D,t):
    # https://en.wikipedia.org/wiki/B%C3%A9zier_curve
    assert 0<=t<=1
    return (1-t)**3*A + 3*(1-t)**2*t*B + 3*(1-t)*t**2*C + t**3*D

# 52, 54 only bezier cureve self-interaction
# 65, 76  only control point segments self-interacted, but not the bezier path.
# 75, 79 both control point segments and bezier path self-interacted
torch.manual_seed(54)

A = torch.rand(2)*100
B = torch.rand(2)*100
C = torch.rand(2)*100
D = torch.rand(2)*100

A= torch.Tensor([56.6840, 64.6931])
B = torch.Tensor([68.0378, 48.7751])
C = torch.Tensor([98.9311, 64.4273])
D = torch.Tensor([32.8432, 82.5911])


X_list=[]
Y_list=[]
for i in range(1,1000):
    X = bezier_point(A,B,C,D, i/1000.1)
    X_list.append(X[0])
    Y_list.append(X[1])
fig, ax = plt.subplots()
ax.plot(X_list, Y_list)

ax.scatter(A[0], A[1])
ax.scatter(B[0], B[1])
ax.scatter(C[0], C[1])
ax.scatter(D[0], D[1])
ax.annotate("A", (A[0], A[1]))
ax.annotate("B", (B[0], B[1]))
ax.annotate("C", (C[0], C[1]))
ax.annotate("D", (D[0], D[1]))
print(A)
print(B)
print(C)
print(D)
plt.show()


