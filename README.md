# Grokking-Phenomenon

We trained a 2-layer Transformer on a modular addition task ($p=97$) and varied the training data fraction ($\alpha$). We observed that lower data fractions lead to exponentially longer generalization delays.
* **$\alpha = 0.3$**: Extreme generalization delay ($\sim$19,500 epochs to grok).
* **$\alpha = 0.5$**: Standard grokking behavior ($\sim$6,300 epochs to grok).
* **$\alpha = 0.8$**: Fast rule extraction ($\sim$1,380 epochs to grok).

image1

Besides, following the framework of [Liu et al. (2022)](#references), we provide a mathematical explanation using the Hessian matrix of the loss landscape. We map the inputs to an embedding space and demonstrate that the model converges when the minimum eigenvalue of the Hessian matrix allows the eigenvectors corresponding to memorization to decay to zero. Our analysis quantitatively explains why the critical sample proportion decreases in higher-dimensional ($n>2$) tasks.

image2
