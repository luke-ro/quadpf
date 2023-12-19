from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

im = Image.open("data/x_150_y_100_km.tiff")
im = np.array(im)
print(im)

cont = -im[0,:]/6

fig,ax = plt.subplots()
x = np.linspace(-250,250,len(cont))
ax.plot(x,cont)
ax.invert_yaxis()

np.save("/home/user/repos/quadpf/data/lunar_contour_slopy.npy",cont)

plt.show()