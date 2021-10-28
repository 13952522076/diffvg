import numpy as np
import matplotlib.pyplot as plt


control_points = "M 226.79583740234375 105.72844696044922 C 215.2116241455078 107.7011947631836 -85.912109375 390.24432373046875 263.56390380859375 149.1533203125 C 233.00003051757812 101.23308563232422 229.15667724609375 101.83340454101562 227.06187438964844 118.47900390625 C 241.30972290039062 103.03022003173828 217.5480194091797 97.00259399414062 -0.13655367493629456 291.5308532714844 C -58.320919036865234 -254.64183044433594 232.82595825195312 62.81068801879883 226.79583740234375 105.72844696044922"
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
