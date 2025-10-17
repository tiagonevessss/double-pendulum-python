import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import sin, cos, pi

l1, l2 = 1, 1
m1, m2 = 1, 1
g = 9.81


def M_inverse(theta):
    t1, t2 = theta
    det_M = m2 * (l1*l2)**2 * (m1 + m2*(1 - cos(t1 - t2)**2))
    return np.array([
        [m2*(l2**2), -m2*l1*l2*cos(t1 - t2)],
        [-m2*l1*l2*cos(t1 - t2), (m1 + m2) * l1**2]
    ]) / det_M


def antecedent_acc(theta, dtheta):
    t1, t2 = theta
    dt1, dt2 = dtheta
    return np.array([
        [-(m1 + m2) * g * l1 * sin(t1) - m2 * l1 * l2 * (dt2**2) * sin(t1 - t2)],
        [-m2 * g * l2 * sin(t2) + m2 * l1 * l2 * (dt1**2) * sin(t1 - t2)]
    ])


dt = 1e-2
tMax = 60
theta = np.array([pi/4, pi/4])
dtheta = np.array([0.0, 0.0])

thetas = []
dthetas = []
for _ in np.arange(0, tMax, dt):
    ddtheta = (M_inverse(theta) @ antecedent_acc(theta, dtheta)).flatten()
    dtheta += ddtheta * dt
    theta += dtheta * dt
    thetas.append(theta.copy())
    dthetas.append(dtheta.copy())

thetas = np.array(thetas)
dthetas = np.array(dthetas)

x1 = l1 * np.sin(thetas[:, 0])
y1 = -l1 * np.cos(thetas[:, 0])
x2 = x1 + l2 * np.sin(thetas[:, 1])
y2 = y1 - l2 * np.cos(thetas[:, 1])


plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.axis('off')
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)

line, = ax.plot([], [], 'o-', lw=2.5, color='#00FFFF', markersize=8)
trail, = ax.plot([], [], '-', lw=1.5, color='orange', alpha=0.6)
time_text = ax.text(-2, 2, '', color='white', fontsize=12)

trail_length = 500
trail_x, trail_y = [], []


def init():
    line.set_data([], [])
    trail.set_data([], [])
    time_text.set_text('')
    return line, trail, time_text


def update(i):
    xdata = [0, x1[i], x2[i]]
    ydata = [0, y1[i], y2[i]]
    line.set_data(xdata, ydata)

    trail_x.append(x2[i])
    trail_y.append(y2[i])
    if len(trail_x) > trail_length:
        trail_x.pop(0)
        trail_y.pop(0)
    trail.set_data(trail_x, trail_y)

    time_text.set_text(f"t = {i*dt:.1f} s")

    line.set_color('#00E5FF')
    # trail.set_alpha(0.4 + 0.4 * np.sin(i / 30))

    return line, trail, time_text


ani = FuncAnimation(fig, update, frames=len(x1),
                    init_func=init, interval=dt, blit=True)

plt.show()

# plt.plot(thetas[:, 0], dthetas[:, 0])
plt.plot(thetas[:, 1], dthetas[:, 1], c="orange")
plt.show()
