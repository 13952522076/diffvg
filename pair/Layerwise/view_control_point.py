import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"


control_points = "394.145 331.428 409.253 325.658 319.737 315.75 270.829 238.104 244.883 249.059 209.444 285.784 206.58 341.234 204.854 412.84 249.637 447.074 284.938 449.045 348.816 456.138 397.825 386.081 394.145 331.428"
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
fig = plt.figure(1, figsize=(4, 3.5))
ax = fig.add_subplot(111)
ax.scatter(x_list, y_list, s=60)
ax.plot(x_list, y_list)
ax.axis('off')

for i, txt in enumerate(labels):
    if i==0:
        ax.annotate(txt, (x_list[i]-5, y_list[i]+5), fontsize=16)
    else:
        ax.annotate(txt, (x_list[i], y_list[i]), fontsize=16)

fig.tight_layout()
fig.savefig("points.pdf", bbox_inches="tight", pad_inches=0, transparent=True)
plt.close()
