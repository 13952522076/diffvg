import numpy as np
import matplotlib.pyplot as plt


control_points = "M 227.87887573242188 103.2909164428711 C 187.25418090820312 146.85670471191406 -113.71832275390625 141.72256469726562 189.20130920410156 314.84271240234375 C 283.3132629394531 114.05476379394531 238.63046264648438 113.10358428955078 240.34561157226562 102.61271667480469 C 228.19638061523438 102.9784927368164 181.31130981445312 144.19061279296875 -110.60543823242188 138.4064178466797 C 243.19680786132812 -129.64764404296875 254.41543579101562 6.276492595672607 227.87887573242188 103.2909164428711"
control_points = control_points.replace("M ", "")
control_points = control_points.replace("C ", "")
control_points = control_points.split(" ")
for i in range(0, len(control_points)):
    control_points[i] = float(control_points[i])
x_list = []
y_list = []
labels = []
for i in range(0, control_points.__len__()//2):
   x_list.append(control_points[2*i])
   y_list.append(-control_points[2*i+1])
   labels.append(i)
print(x_list)
print(y_list)
print(control_points)
print(control_points.__len__())
fig, ax = plt.subplots()
ax.scatter(x_list, y_list)
ax.plot(x_list, y_list)


for i, txt in enumerate(labels):
    ax.annotate(txt, (x_list[i], y_list[i]))

plt.show()
