import numpy as np
import matplotlib.pyplot as plt


control_points = "M 179.10841369628906 140.6649932861328 C 205.34938049316406 149.47314453125 140.47564697265625 172.563232421875 118.27446746826172 169.12921142578125 C 136.96881103515625 168.95640563964844 173.2473602294922 141.53358459472656 156.56124877929688 151.6849822998047 C 79.07022094726562 175.21127319335938 54.5128059387207 119.72750854492188 55.33195495605469 150.5403289794922 C 128.1562042236328 188.63267517089844 135.57015991210938 160.23944091796875 179.10841369628906 140.6649932861328"
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
