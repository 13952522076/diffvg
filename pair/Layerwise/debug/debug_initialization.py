import numpy as np
import cv2
import random
import torch
import matplotlib.pyplot as plt

def get_bezier_circle(radius=1, segments=4, bias=None):
    points = []
    if bias is None:
        bias = (random.random(), random.random())
    avg_degree = 360 / (segments*3)
    for i in range(0, segments*3):
        point = (np.cos(np.deg2rad(i * avg_degree)),
                    np.sin(np.deg2rad(i * avg_degree)))
        points.append(point)
    points = torch.tensor(points)
    points = (points)*radius + torch.tensor(bias).unsqueeze(dim=0)
    points = points.type(torch.FloatTensor)
    return points


if __name__ == "__main__":
    points = get_bezier_circle(0.1, 8)
    print(points.shape)
    print(points)

    x_list = []
    y_list = []
    labels = []
    for i in range(0, points.shape[0]):
        x_list.append((points[i,0]).item())
        y_list.append((points[i,1]).item())
        labels.append(i)
    fig, ax = plt.subplots()
    ax.scatter(x_list, y_list)
    ax.plot(x_list, y_list)

    for i, txt in enumerate(labels):
        ax.annotate(txt, (x_list[i], y_list[i]))

    plt.show()
