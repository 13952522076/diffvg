import torch
import numpy as np


def cross_mul(a, b, c):
    return (c[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(c[0]-a[0])


def xing_loss(x_list, scale=1.0):  # x[ npoints,2]
    loss = 0.
    for x in x_list:
        x1_first = x[:1, :]
        x1_rest = x[1:, :]
        x1 = torch.cat([x1_rest, x1_first], dim=0)
        x = torch.cat([x,x1],dim=1)
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[0]):
                if i==j:
                    continue
                q1x = x[i,0]
                q2x = x[i,2]
                q1y = x[i,1]
                q2y = x[i,3]
                p1x = x[j,0]
                p2x = x[j,2]
                p1y = x[j,1]
                p2y = x[j,3]
                if q1x==q2x and q1y==q2y:
                    continue
                if p1x==p2x and p1y==p2y:
                    continue
                if p1x==q1x and p1y==q1y and p2x==q2x and p2y==q2y:
                    continue
                if p1x==q2x and p1y==q2y and p2x==q1x and p2y==q1y:
                    continue

                A = np.array([p1x, p1y])
                B = np.array([p2x, p2y])
                C = np.array([q1x, q1y])
                D = np.array([q2x, q2y])

                d1 = cross_mul(A, C, D)
                d2 = cross_mul(B, C, D)
                d3 = cross_mul(A, B, C)
                d4 = cross_mul(A, B, D)
                value = (d1*d2 + d3*d4)
                loss+=value

    return loss/(len(x_list))



if __name__ == "__main__":
    x = torch.rand([14,2])
    scale = 0.001
    y = xing_loss([x], scale)
    print(y)

