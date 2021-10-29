import numpy as np
import matplotlib.pyplot as plt


control_points = "M 274.88165283203125 113.8409423828125 C 232.37652587890625 137.7791748046875 133.2840118408203 207.431884765625 171.60293579101562 317.98101806640625 C 350.1929931640625 144.4225311279297 221.87977600097656 93.48306274414062 200.5298614501953 186.056396484375 C 99.17528533935547 294.9606018066406 -29.997480392456055 388.7394104003906 -26.232616424560547 2.4835691452026367 C 310.3525085449219 -23.13043212890625 275.82281494140625 -4.1869587898254395 274.88165283203125 113.8409423828125"
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
