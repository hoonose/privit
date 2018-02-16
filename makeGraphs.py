import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import functions as f

upper_bound = 9000
interpolate_points = 20
colors = ["b", "0.0", "y", "c"]
alpha=0.1
epsilon=0.1

fig = plt.figure()
ax = fig.add_subplot(111)
fig.suptitle('Sample Complexities for Uniformity Testing', fontsize=14, fontweight='bold')

with open("outputs/PrivIT_alpha%.2f_epsilon%.2f.txt"%(alpha, epsilon)) as lo_file:
	c = lo_file.readlines()

c = [x.strip() for x in c]
c = [[int(y.split(": ")[1]) for y in x.split(", ")] for x in c]
c = np.array([[x[0], max(x[1], x[2]), x[3]] for x in c])
c = c[c[:,0]<upper_bound]
c = np.transpose(c)

# p_x = range(min(c_x), upper_bound, 10)
# p_y = [f.getUniformDPBound(n, alpha, epsilon) for n in p_x]

# xnew = np.linspace(c_x.min(), c_x.max(), interpolate_points)
# o_corr, = ax.plot(xnew, spline(c_x, c_y, xnew), label = "Priv'IT(Correctness)", linewidth = 2.0, color = colors[0])
# o_priv, = ax.plot(xnew, spline(c_x, p_y, xnew), label = "Priv'IT(Privacy)", linewidth = 2.0, color = colors[1])
o_corr, = ax.plot(c[0], c[1], label = "Priv'IT(Correctness)", linewidth = 2.0, color = colors[0])
o_priv, = ax.plot(c[0], c[2], label = "Priv'IT(Privacy)", linewidth = 2.0, color = colors[1])

with open("outputs/KR_alpha%.2f_epsilon%.2f.txt"%(alpha, epsilon)) as r_file:
    c_r = r_file.readlines()
c_r = [x.strip() for x in c_r]
c_r = np.array([[int(y.split(": ")[1]) for y in x.split(", ")] for x in c_r])
c_r = c_r[c_r[:,0]<=upper_bound]
c_r = np.transpose(c_r)

# xnew = np.linspace(c_r[0].min(), c_r[0].max(), interpolate_points)
# r_corr, = ax.plot(xnew, spline(c_r[0], c_r[1], xnew), label = "zCDP-GOF", linewidth = 2.0, color = colors[2])
r_corr, = ax.plot(c_r[0], c_r[1], label = "zCDP-GOF", linewidth = 2.0, color = colors[2])

with open("outputs/GLRV_alpha%.2f_epsilon%.2f.txt"%(alpha, epsilon)) as s_file:
    c_s = s_file.readlines()
c_s = [x.strip() for x in c_s]
c_s = np.array([[int(y.split(": ")[1]) for y in x.split(", ")] for x in c_s])
c_s = c_s[c_s[:,0]<=upper_bound]
c_s = np.transpose(c_s)

# xnew = np.linspace(c_s[0].min(), c_s[0].max(), interpolate_points)
# s_corr, = plt.plot(xnew, spline(c_s[0], c_s[1], xnew), label = "MCGOF", linewidth = 2.0, color = colors[3])
s_corr, = plt.plot(c_s[0], c_s[1], label = "MCGOF", linewidth = 2.0, color = colors[3])

ax.legend(loc='upper left', shadow=True, fontsize='medium')
ax.set_xlabel('Support Size (n)')
ax.set_ylabel('Sample Complexity (m)')
plt.savefig("plots/plot_alpha%.2f_epsilon%.2f.pdf"%(alpha, epsilon))

