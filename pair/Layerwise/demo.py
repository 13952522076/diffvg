import numpy as np
import torch
import random
import matplotlib.pyplot as plt
# points = []
# pts = [(np.cos(i*np.pi), np.cos(i*np.pi)) for i in np.linspace(0,2,num=4,endpoint=False)]
# for (w0, h0), (w1, h1) in zip(pts[:-1], pts[1:]):
#     points.append((w0, h0))
#     points.append((w0+1/3*(w1-w0), h0+1/3*(h1-h0)))
#     points.append((w0+2/3*(w1-w0), h0+2/3*(h1-h0)))
#
# points.append(pts[-1])
# points = torch.tensor(points)
# print(f"\n\n\nthe points are: {points}\n\n\n")
# print(points.shape)


def get_bezier_circle(radius=1, segments=4, bias=None):
    points = []
    if bias is None:
        bias = (random.random(), random.random())
    m_point = (1.0, 0.0)
    points.append(m_point)
    m1_point_angle = np.degrees(4/3*np.tan(np.pi/(2*segments)))
    m1_point_radius = np.sqrt(1+ ( 4/3*np.tan(np.pi/(2*segments)))**2)
    m3_point_angle = 360/segments
    for i in range(0, segments):
        m1_point = (m1_point_radius*np.cos(np.deg2rad(i*m3_point_angle+m1_point_angle)),
                    m1_point_radius*np.sin(np.deg2rad(i*m3_point_angle+m1_point_angle)))
        m2_point_angle = m3_point_angle-m1_point_angle
        m2_point = (m1_point_radius*np.cos(np.deg2rad(i*m3_point_angle+m2_point_angle)),
                    m1_point_radius*np.sin(np.deg2rad(i*m3_point_angle+m2_point_angle)))
        m3_point = (np.cos(np.deg2rad(i*m3_point_angle+m3_point_angle)),
                    np.sin(np.deg2rad(i*m3_point_angle+m3_point_angle)))
        points.append(m1_point)
        points.append(m2_point)
        points.append(m3_point)
    points = torch.tensor(points)
    points = points*radius+torch.tensor(bias).unsqueeze(dim=0)
    print(points.shape)
    return points

    print(points.shape)

points = get_bezier_circle(radius=0.05, segments=4, bias=(random.random(), random.random()))


points = points.numpy()

plt.plot(points[:,0],points[:,1], 'o',color='b')

plt.show()
print(points)
