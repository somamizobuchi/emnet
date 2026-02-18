"""Quick script to visualize the raised cosine basis."""

import numpy as np
import matplotlib.pyplot as plt
from src.utils.constraints import make_raised_cosine_basis

n_taps = 16
n_basis = 6
log_offset = 1.0

basis = make_raised_cosine_basis(
    n_taps, n_basis, log_offset
).numpy()  # (n_taps, n_basis)

weights = np.random.randn(n_basis)
kernel = basis @ weights

lags = np.arange(n_taps - 1, -1, -1)  # n_taps-1 (oldest) down to 0 (most recent)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

for j in range(n_basis):
    ax1.plot(lags, basis[:, j], label=f"basis {j}")
ax1.set_xlabel("Lag (samples, 0 = most recent)")
ax1.set_ylabel("Weight")
ax1.set_title(
    f"Raised cosine basis  (n_taps={n_taps}, n_basis={n_basis}, log_offset={log_offset})"
)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2.plot(lags, kernel, color="black")
ax2.set_xlabel("Lag (samples, 0 = most recent)")
ax2.set_ylabel("Weight")
ax2.set_title(f"Example kernel  (weights={np.round(weights, 2)})")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("rcb.png", dpi=150)
plt.show()
print("Saved rcb.png")
