import numpy as np
import matplotlib.pyplot as plt


control_points = "M 242.0737762451172 105.86231231689453 C 213.0720977783203 176.3149871826172 50.9930419921875 297.4375915527344 235.99131774902344 259.1036071777344 C 292.2555236816406 20.933101654052734 238.76405334472656 115.16631317138672 135.5102081298828 256.0452575683594 C -6.474006652832031 299.5086975097656 -11.899201393127441 307.558349609375 -12.017705917358398 0.647771418094635 C 331.8502197265625 -17.864585876464844 244.94972229003906 -10.443171501159668 242.0737762451172 105.86231231689453"
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
