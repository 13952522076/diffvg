import numpy as np
import matplotlib.pyplot as plt


control_points = "M 285.1794738769531 19.067455291748047 C 239.795654296875 229.55426025390625 148.92568969726562 255.4611053466797 235.6516571044922 258.909423828125 C 276.72589111328125 69.71878051757812 267.2552490234375 108.81698608398438 184.22523498535156 269.32012939453125 C -0.9490314722061157 294.3135681152344 -15.397896766662598 325.2052001953125 -14.558411598205566 1.3756033182144165 C 200.23036193847656 -18.2019100189209 299.33343505859375 -12.812262535095215 285.1794738769531 19.067455291748047"
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
