## 1. Filter Energy Constraint (Unit Norm)

The model enforces a hard constraint on the spatiotemporal filters to ensure they only represent the pattern of the input, not the magnitude.

**Mechanism:** Hard L2-normalization ($||W||_2 = 1$).
  - **Spatial:** The weights are projected back onto a unit sphere after every gradient step (`self.W /= self.W.norm()`).
  - **Temporal:** The kernels are normalized on-the-fly during the forward pass.

**Purpose:** It decouples the feature selection (the filter shape) from the sensitivity (the gain). This prevents the model from "cheating" the noise by simply scaling up filter weights; instead, it must find the most informative shape.

## 2. Firing Rate Constraint (Metabolic Budget)

The model enforces an equality constraint on the output activity to ensure the average firing rate $\bar{r}$ matches a specific target (default 1.0).

**Mechanism:** Augmented Lagrangian Method (ALM). It adds two penalty terms to the loss function:
  1. **Linear Penalty** ($\lambda \cdot h$): A learnable Lagrange multiplier ($\lambda$) that provides a directional force.
  2. **Quadratic Penalty** ($\frac{\rho}{2} \cdot h^2$): A squared error term that ensures optimization stability.

**Optimization:** The model uses dual-ascent: it performs gradient descent on the weights to minimize error, but gradient ascent on $\lambda$ to satisfy the constraint.

**Purpose:** This represents a metabolic bottleneck. It forces the model to be "efficient"—to pick filters and gains that capture the most information possible without exceeding a fixed spike budget.

## Summary Table

| Constraint | Target | Implementation | Conceptual Role |
| :--- | :--- | :--- | :--- |
| Filter Energy | $\|\|W\|\|_2 = 1$ | Hard Normalization | Hardware: Limits the "power" of the physical sensor. |
| Firing Rate | $E[r] = \text{target}$ | Augmented Lagrangian | Metabolism: Limits the "cost" of the neural signal. |

In short: the Filter Energy constraint forces the filters to have the right shape, while the Firing Rate constraint forces the Gain (logA) to be as informative as possible within a restricted budget.
