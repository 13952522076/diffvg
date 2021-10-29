import numpy as np
import matplotlib.pyplot as plt


control_points = "M 256.3534851074219 110.7645034790039 C 71.21488952636719 181.9011688232422 -10.244577407836914 324.7065124511719 237.1314697265625 284.7209777832031 C 262.41851806640625 89.2352066040039 245.9853057861328 118.39578247070312 187.95252990722656 141.21246337890625 C 145.94461059570312 167.42044067382812 115.1034927368164 246.79754638671875 -58.48623275756836 237.8876495361328 C -52.845550537109375 -154.3301239013672 289.62677001953125 -114.38773345947266 256.3534851074219 110.7645034790039"
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
