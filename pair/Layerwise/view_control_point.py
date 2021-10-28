import numpy as np
import matplotlib.pyplot as plt


control_points = "M 231.2699737548828 102.01410675048828 C 215.86343383789062 121.92900848388672 91.79105377197266 207.20013427734375 19.3505802154541 250.4076690673828 C 440.82666015625 278.14208984375 239.76922607421875 116.5105972290039 233.24356079101562 104.46342468261719 C 157.57357788085938 168.86863708496094 -7.877041339874268 351.5126953125 -41.921146392822266 13.529959678649902 C 248.0608673095703 -59.36091995239258 316.05645751953125 3.7208786010742188 231.2699737548828 102.01410675048828"
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
