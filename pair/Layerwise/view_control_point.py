import numpy as np
import matplotlib.pyplot as plt


control_points = "M 225.0663299560547 106.76058959960938 C 212.41659545898438 98.1502456665039 -128.12525939941406 58.59368896484375 166.21095275878906 327.2657775878906 C 254.1380157470703 109.2364730834961 240.31613159179688 112.6659164428711 233.05340576171875 113.32968139648438 C 236.4588165283203 99.80708312988281 226.33973693847656 100.40277099609375 77.80248260498047 66.88341522216797 C 304.624755859375 -168.0904083251953 228.4355926513672 88.4002914428711 225.0663299560547 106.76058959960938"
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
