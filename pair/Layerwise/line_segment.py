import os
import numpy as np

import matplotlib.pyplot as plt
# https://www.cnblogs.com/Duahanlang/archive/2013/05/11/3073434.html this one is bullshit!

# use this one:
# https://stackoverf ow.com/questions/3838329/how-can-i-check-if-two-segments-intersect


move=5

try:
    os.makedirs("line_segment")
except OSError as exc:  # Python >2.5
        pass

def cross_mul(a, b, c):
    return (c[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(c[0]-a[0])


for p1x in range(1,move):
    for p1y in range(1,move):
        print(f"\n\n\n====== {p1x}-{p1y} ======\n\n\n")
        for p2x in range(1,move):
            for p2y in range(1,move):
                for q1x in range(1,move):
                    for q1y in range(1,move):
                        for q2x in range(1,move):
                            for q2y in range(1,move):
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

                                value1 = 0
                                value2 = 0
                                if d1*d2<0 and d3*d4<0:
                                    value1 = d1*d2*d3*d4
                                    value2 = d1*d2 + d3*d4
                                    print(value1, value2)
                                    fig, ax1 = plt.subplots()
                                    # ax1.plot(x, y)
                                    # ax2.plot(x, -y)
                                    # ax1.xlim(0, move+1)
                                    # ax1.ylim(0, move+1)
                                    # ax1.scatter([p1x, p1y], [p2x, p2y])
                                    # ax1.scatter([q1x, q2x], [q1y, q2y])
                                    print(f"{p1x}{p1y}, {p2x}{p2y} | {q1x}{q1y}, {q2x}{q2y}")
                                    ax1.plot([p1x, p2x], [p1y, p2y])
                                    ax1.plot([q1x, q2x], [q1y, q2y])
                                    # ax2.bar(range(1), [value1])
                                    # ax2.bar(range(1), [-value2])
                                    plt.title(f"{value1} | {value2}")
                                    plt.show()
                                    # plt.close()
                                    # fig.close()
                                    print(1)










