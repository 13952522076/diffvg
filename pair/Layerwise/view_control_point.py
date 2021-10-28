import numpy as np
import matplotlib.pyplot as plt


control_points = "M 225.1237335205078 106.68374633789062 C 212.37725830078125 98.14036560058594 -127.70963287353516 60.138065338134766 168.56797790527344 327.2427978515625 C 254.0561981201172 109.21398162841797 240.29615783691406 112.66291809082031 233.0505828857422 113.35892486572266 C 237.85128784179688 99.84041595458984 226.1619873046875 100.94896697998047 76.10830688476562 67.40980529785156 C 305.0262145996094 -169.02862548828125 228.33282470703125 88.07673645019531 225.1237335205078 106.68374633789062"
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
