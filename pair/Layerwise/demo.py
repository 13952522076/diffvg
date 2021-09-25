import numpy as np
import torch
import random
import matplotlib.pyplot as plt
points = []
p0 = (random.random(), random.random())
points.append(p0)
for j in range(4):
    radius = 0.05
    p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
    p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
    p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
    points.append(p1)
    points.append(p2)
    if j < 4 - 1:
        points.append(p3)
        p0 = p3
points = torch.tensor(points)
print(points.shape)
print(points)
print(points.dtype)


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
    points.reverse()
    points = torch.tensor(points)
    points = points[:-1,:]
    points += 1
    points = (points)*radius + torch.tensor(bias).unsqueeze(dim=0)
    points = points.type(torch.FloatTensor)
    return points

points = get_bezier_circle(radius=0.03, segments=8, bias=None)
print(points.shape)
print(points)
print(points.dtype)


points = points*240

points = points.numpy()

# plt.plot(points[:,0],points[:,1], 'o',color='b')
# for i in range(0, len(points[:,0])):
#     plt.annotate(i+1, (points[i,0], points[i,1]))
#
# plt.show()
# print(points)

file = open('demodemo.svg', 'w')
file.write('<?xml version="1.0" ?>\n')
file.write('<svg height="240" version="1.1" width="240" xmlns="http://www.w3.org/2000/svg">\n')
file.write('<defs/>\n')
file.write('<g>\n')
path_str = '<path d="M '
path_str+=f'{points[0][0]} {points[0][1]} '
for i in range(1, len(points)):
    if i%3==1:
       path_str+='C '
    path_str+=f'{points[i][0]} {points[i][1]} '
path_str+=f'{points[0][0]} {points[0][1]} '
path_str += '" fill="rgb(106, 255, 255)" opacity="1.0" stroke-width="2.0"/>'
file.write(f'{path_str}\n')
file.write('</g>\n')
file.write('</svg>')
file.close()

