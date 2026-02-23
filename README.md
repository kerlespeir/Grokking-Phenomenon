# Grokking-Phenomenon

We trained a 2-layer Transformer on a modular addition task ($p=97$) and varied the training data fraction ($\alpha$). We observed that lower data fractions lead to exponentially longer generalization delays.
* **$\alpha = 0.3$**: Extreme generalization delay
* **$\alpha = 0.5$**: Standard grokking behavior
* **$\alpha = 0.8$**: Fast rule extraction

![png](grokking_alpha.png)

Besides, following the framework of [Liu et al. (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/dfc310e81992d2e4cedc09ac47eff13e-Abstract-Conference.html), we provide a mathematical explanation using the Hessian matrix of the loss landscape. We map the inputs to an embedding space and demonstrate that the model converges when the minimum eigenvalue of the Hessian matrix allows the eigenvectors corresponding to memorization to decay to zero. Our analysis quantitatively explains why the critical sample proportion decreases in higher-dimensional ($n>2$) tasks.

![png](theoryofgrokking.png)
