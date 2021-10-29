import numpy as np
import matplotlib.pyplot as plt


control_points = "M 247.16319274902344 118.17431640625 C 221.7051544189453 150.60641479492188 179.5376434326172 336.55535888671875 276.26361083984375 164.89352416992188 C 238.7407989501953 134.0352020263672 218.95289611816406 106.02735137939453 237.62646484375 132.8995819091797 C 223.22425842285156 114.90709686279297 168.1798095703125 356.3849182128906 -49.485408782958984 214.60043334960938 C -51.706695556640625 -154.3482208251953 301.14208984375 -121.48951721191406 247.16319274902344 118.17431640625"
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
