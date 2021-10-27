import numpy as np
import matplotlib.pyplot as plt


control_points = "M 235.61546325683594 101.95073699951172 C 225.90269470214844 121.75647735595703 49.818607330322266 169.98204040527344 114.31741333007812 328.824462890625 C 342.5984191894531 144.75607299804688 232.38381958007812 108.08136749267578 222.35775756835938 115.7666244506836 C 171.31982421875 134.534423828125 66.15557861328125 162.65435791015625 -56.444400787353516 12.645215034484863 C 308.0386657714844 -35.32614517211914 287.6276550292969 28.352392196655273 235.61546325683594 101.95073699951172"
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
