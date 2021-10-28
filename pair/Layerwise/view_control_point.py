import numpy as np
import matplotlib.pyplot as plt


control_points = "M 239.49972534179688 157.1817169189453 C 113.30311584472656 118.89351654052734 85.03292846679688 237.01612854003906 33.36125946044922 285.0933532714844 C 432.9490051269531 234.27957153320312 229.3842010498047 121.79268646240234 167.55894470214844 155.44723510742188 C 190.25836181640625 139.2013702392578 35.29796600341797 275.44781494140625 -59.25229263305664 60.343406677246094 C 131.10617065429688 -109.03233337402344 321.8238830566406 -46.71402359008789 239.49972534179688 157.1817169189453"
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
