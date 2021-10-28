import numpy as np
import matplotlib.pyplot as plt


control_points = "M 222.3904266357422 103.57666778564453 C -25.862876892089844 67.8792495727539 -92.73348999023438 59.857757568359375 40.708213806152344 320.8557434082031 C 406.9229431152344 222.84707641601562 246.4051055908203 115.68431854248047 240.97080993652344 80.91796875 C 217.29092407226562 119.9404067993164 225.1883087158203 100.74723052978516 -69.9239273071289 61.21849060058594 C 259.35308837890625 -179.25155639648438 270.0983581542969 41.283966064453125 222.3904266357422 103.57666778564453"
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
