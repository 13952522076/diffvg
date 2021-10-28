import numpy as np
import matplotlib.pyplot as plt


control_points = "M 225.14529418945312 106.70245361328125 C 212.4116668701172 98.15098571777344 -127.82332611083984 59.630775451660156 167.16152954101562 326.7458801269531 C 253.9549560546875 109.22181701660156 240.28506469726562 112.66492462158203 233.087158203125 113.34879302978516 C 237.3100128173828 99.83097076416016 226.2572784423828 100.8985824584961 74.53778076171875 67.62417602539062 C 305.16583251953125 -169.65890502929688 228.38784790039062 88.16215515136719 225.14529418945312 106.70245361328125"
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
