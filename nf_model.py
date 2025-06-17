import numpy as np
import matplotlib.pyplot as plt

# 1. Input-Verteilung: 2D-Gitter
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
u1, u2 = np.meshgrid(x, y)
u = np.stack([u1.ravel(), u2.ravel()], axis=1)

# 2. Definiere s(u1) und t(u1) — wie ein kleines Netz
def s(u1): return 0.5 * np.tanh(u1)
def t(u1): return 1.0 * np.sin(u1)

# 3. Affine Coupling Transformation
v1 = u[:, 0]
v2 = u[:, 1] * np.exp(s(u[:, 0])) + t(u[:, 0])
v = np.stack([v1, v2], axis=1)

# Dein Plot (vereinfachtes Beispiel)
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1])

# Zeichne den Canvas
fig.canvas.draw()

# Bilddaten als RGBA-Array holen
img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))

print(img.shape)  # z.B. (480, 640, 4)