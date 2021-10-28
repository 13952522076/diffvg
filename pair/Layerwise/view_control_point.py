import numpy as np
import matplotlib.pyplot as plt


control_points = "M 248.74603271484375 -16.22156524658203 C 253.31298828125 -14.292258262634277 249.21421813964844 -17.835498809814453 -24.816747665405273 1.7685065269470215 C -17.633852005004883 204.4075164794922 -2.411907911300659 253.61085510253906 29.538692474365234 300.03271484375 C 170.1244354248047 260.470947265625 211.1136016845703 253.77560424804688 329.99139404296875 232.62106323242188 C 266.48199462890625 169.44915771484375 251.51551818847656 155.7205810546875 248.74603271484375 -16.22156524658203"
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
