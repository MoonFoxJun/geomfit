# Kernel ↔ Convolution ↔ Attention: Three Views of the Same Machine

> Reading time: 10 minutes. Written for the "learn-by-doing" Functional Solver author.
> Bottom line: **Convolution is a translation-invariant fixed kernel; kernel methods are a fixed (or tunable-metric) kernel; attention is a data-adaptive kernel.** They are not three different things — they are three configurations of the same thing.

---

## 0. The one-sentence version

Think of "the weight of the contribution between two data points" as a function k(x, y); the three methods differ only in what k looks like:

| Method | Form of the kernel | Source of weights | Content-dependent? |
|---|---|---|---|
| Convolution | k(x,y) = w(x−y) | Fixed, depends only on displacement | ❌ |
| Classical kernel methods (this library) | k(x,y) = RBF/Matern/Mahalanobis… | Fixed or with a tunable metric | ❌ (metric can be learned) |
| Attention | k(xᵢ,xⱼ) = softmax(⟨W_q xᵢ, W_k xⱼ⟩/√d) | Determined jointly by all inputs | ✅ |

---

## 1. Convolution is a kernel operator (mathematically)

Convolution is defined as a **translation-invariant integral kernel operator**:

$$(f*w)(x) = \int_\Omega f(y)\, w(x-y)\, dy,\qquad k(x,y)=w(x-y)$$

So the phrase "replacing convolution with a kernel operator" is not quite accurate — convolution **already is** a kernel operator. What you actually want to replace is the **fixed, content-independent, translation-invariant kernel**.

### The three costs of convolution
1. **Content-independent weights**: scanning a cat's face and a patch of fur uses the same W;
2. **Fixed receptive field**: long-range dependencies must propagate through many layers;
3. **Translation invariance is a strong assumption**: positional information itself is flattened (which buys parameter sharing and data efficiency).

---

## 2. Classical kernel methods (what your KernelSolver does)

- Pick a positive-definite kernel k(x, y) (RBF/Matern/polynomial/Mahalanobis…); it implicitly defines a function space (an RKHS);
- **Representer theorem**: the optimal function is always a superposition of "kernel bumps" centered at the data points:

$$f(x) = \sum_i \alpha_i\, k(x, x_i),\qquad \text{solution } (K+\alpha I)\,\alpha = y$$

- RKHS norm: ‖f‖_H = √(αᵀKα) — this is exactly what `KernelSolver.compute_function_norm` computes.

**Costs**: the kernel itself is fixed (a smoothness prior — if the wrong one is chosen, the fit suffers); O(n²)/O(n³) does not scale; features cannot be learned automatically.

---

## 3. Attention = a data-adaptive kernel

Output of one self-attention layer:

$$A_{ij} = \mathrm{softmax}\!\left(\frac{\langle W_q x_i,\ W_k x_j\rangle}{\sqrt d}\right),\qquad y_i = \sum_j A_{ij}\, V_j$$

- **When W_q = W_k = I and V = y**:

$$y_i = \sum_j \mathrm{softmax}\!\left(\frac{\langle x_i, x_j\rangle}{\sqrt d}\right) y_j$$

This is exactly **Nadaraya-Watson kernel smoothing** (the simplest "attention", known since 1964): the output is a kernel-weighted average of the target values, with weights normalized row by row. It is isomorphic to the representer theorem from Section 2 — the only difference is that "solving the linear system (K+αI)α=y" is replaced by "row-wise softmax".

- **Learning W_q and W_k = learning a kernel**; the weights A_ij are determined by **all inputs jointly** → content-adaptive, with context injected directly at every layer.

**Costs**: O(n²); the translation prior is lost (positional encodings are needed to compensate); data-hungry (small datasets overfit easily).

---

## 4. A precise version of your intuition

> "Some data points contribute less than they should, others more."

Translated into mathematics: **a fixed kernel cannot reweight by content.** A "paw" that only makes sense next to a cat's body is averaged with fixed weights in isolation (under-contributing), while redundant background is weighted equally with other regions (over-contributing). Attention fixes this with weights "determined by all inputs".

The first step already present in this library: `MahalanobisKernel` estimates its metric from the data automatically — this is the seed of "making the kernel data-aware". The next step (only if you want to pursue it): a metric that varies with position (a locally adaptive kernel).

---

## 5. Names along this line of research (for future reference)

- **NTK** (Neural Tangent Kernel): in the infinite-width limit, neural networks reduce to kernel regression — deep learning "converges" to kernel methods;
- **Attention is a kernel** (Tsai et al., 2019): writes transformer attention rigorously as kernel smoothing;
- **Deformable Convolution**: lets the receptive field deform along the structure instead of staying a rigid rectangle;
- **Nadaraya-Watson estimator** (1964): the ancestor of kernel smoothing.

---

## 6. When to use which (one table)

| Method | Complexity | Prior | Best for |
|---|---|---|---|
| Convolution | Linear | Translation invariance (strong) | Small data, images, strong-prior tasks |
| Classical kernel (this library) | Quadratic/cubic | Smoothness (tunable metric) | Small-sample function approximation |
| Attention | Quadratic | Almost none (needs positional encoding) | Large data, long-range dependencies |

Engineering reality: low-level convolution (exploit the prior) + high-level attention (exploit adaptivity), or keep kernel machines for small data.

---

## 7. The counterparts in this library

- `InnerProduct` + Gram matrix = the "inner-product view" of a kernel matrix;
- `KernelSolver` (representer theorem + RKHS norm) = the machine from Section 2;
- `MahalanobisKernel` = a kernel that is aware of data correlation;
- The demo script `examples/demo_attention_as_kernel.py` = an isomorphism comparison between this library's kernel regression and a "fixed-weight attention"; run it and compare the two curves.
