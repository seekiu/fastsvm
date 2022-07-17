# Summary of original SMO algorithm by Platt

The SMO algorithm is originally proposed by Platt in the paper "Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines" published in 1998. As a first step and baseline for the FastSVM library, we build an SVM solver based on the original SMO algorithm. In this document, I summarize the key ideas in the SMO algorithm.

The goal of SMO is to solve the dual formulation of SVM, in the following form:

$$
\begin{aligned}
\max_{\alpha} \quad & \sum_{i=1}^m \alpha_i -
    \frac{1}{2} \sum_{i=1}^m\sum_{j=1}^m \alpha_i\alpha_jy_iy_j
    K(x_i, x_j) \\
\text{s.t.} \quad & \sum_{i=1}^m \alpha_i y_i = 0 \\
    & 0 \leq \alpha_i \leq C, \quad i = 1, 2, \ldots, m
\end{aligned}
$$

There are $m$ optimization variables (Lagrange multiplier for the dual problem), equal to the number of training samples. SMO is an efficient algorithm specifically to solve the above optimization problem.

On a high-level, the basic idea of SMO algorithm consists of 2 parts:

1. Given 2 selected  $\alpha$-s, keep all other $\alpha$ constant, the optimization becomes a quadratic problem for single variable (because of the equality constraint). This subproblem has a closed-form solution.
1. A loop that chooses 2 $\alpha$ heuristically. The idea is to choose the ones that most likely violate the KKT condition and violate the most.


## Solving subproblem of 2 selected Lagrange multipliers

If we select only 2 $\alpha$ and consider others constant, the optimization problem reduces to a much simpler subproblem: A quadratic optimization with 2 variables, 1 equality constraint and 2 inequality constraints. Because of the equality constraint, the subproblem is essentially a single-variablem quadratic problem. It has a closed-form analytic solution.

Here we don't go through the whole process of solving the problem, just show the solved equations.

For simplicity, we note the selected $\alpha$ as $\alpha_1$ and $\alpha_2$, the we update the values based on the following:

$$
\alpha_2^\text{new, unclipped} = \alpha_2 + \frac{y_2(E_1 - E_2)}{\eta}
$$

where $\eta = K_{11} + K_{22} - 2K_{12}$, and $K_{ij}$ is short for $K(x_i, x_j)$.

Then the new value needs to be clipped to satisfy the inequality constraint:

$$
\alpha_2^\text{new} = \text{clip}(\alpha_2^\text{new, unclipped}, L, H)
$$

The bounds for the clip function can be calculated as:

$$
\begin{cases}
\text{if } y_1 \neq y_2:
L = \max(0, \alpha_2^text{old} - \alpha_1^\text{old}),
H = \min(C, C+ \alpha_2^\text{old} - \alpha_1^\text{old}), \\
\text{if } y_1 = y_2:
L = \max(0, \alpha_2^\text{old} + \alpha_1^\text{old} - C),
H = \min(C, \alpha_2^\text{old} + \alpha_1^\text{old})
\end{cases}
$$

Then the updated value of the other $\alpha$ can be computed by:

$$ \alpha_1^\text{new} = \alpha_1 + s(\alpha_2 - \alpha_2^\text{new}) $$

where $s = y_1y_2$.

Under some unusually circumstances, the $\eta$ may not be a positive value. In this case, the $\alpha_2^\text{new}$ should be chosen at one of the bounds, whichever gives lower value for the objective function. The objective function at the bounds can be calculated as:

$$
\begin{aligned}
\Psi_L & =  L_1f_1 + Lf_2 + \frac{1}{2}L_1^2K_{11} + \frac{1}{2}L^2K_{22} + sLL_1K_{12} \\
\Psi_H & =  H_1f_1 + Hf_2 + \frac{1}{2}H_1^2K_{11} + \frac{1}{2}H^2K_{22} + sHH_1K_{12}
\end{aligned}
$$

where $f_1 = y_1(E_1+b) - \alpha_1K_{11} - s\alpha_2K_{12}$, 
$f_2 = y_2(E_2+b) - s\alpha_1K_{12} - \alpha_2K_{22}$, 
$L_1 = \alpha_1 + s(\alpha_2 - L)$, $H_1 = \alpha_1 + s(\alpha_2 - H)$.


## Heuristics for choosing the Lagrange multipliers

There are 2 Lagrange multipliers to be selected. This can be implemented as an outer loop that chooses one multipler and an inner loop that chooses another.

* For the first multiplier:
    * Loop all multipliers (in the paper also mentions: find the ones that violates KKT condition, but in fact it's not in the code)
    * Loop all non-bound multipliers (those most likely to violate KKT condition), and find then ones that actually violates KKT condition.
* For the second multiplier:
    * Use $\alpha_2$ that maximizes $|E_1-E_2|$, which leads to the biggest update step. Here the $E$ is super compute-heavy and we should use cached version.
    * If the selected $\alpha_2$ can make useful progress for the given $\alpha_1$, then:
        * Loop all the non-bound $\alpha_2$, starting at random point
        * Loop all the multipliers, starting at random point


## Computing the threshold $b$

The threshold $b$ should be update in each iteration with the following:

$$
\begin{aligned}
b_1 & = E_1 + y_1(\alpha_1^\text{new}-\alpha_1)K_{11} + y_2(\alpha_2^\text{new}-\alpha_2)K_{12} + b \\
b_2 & = E_2 + y_1(\alpha_1^\text{new}-\alpha_1)K_{12} + y_2(\alpha_2^\text{new}-\alpha_2)K_{22} + b
\end{aligned}
$$

and update threshold with $b = (b_1 + b_2) / 2$.


## Pseudo code for SMO

The key process can be expressed with the following pseudo code:

```python
def main():
    num_changed = 0
    example_all = 1
    while num_changed > 0 or examine_all:
        num_changed = 0
        if examine_all:
            for i1 in range(m):  ## loop all alpha
                num_changed += examine_example(i1)
        else:
            for i1 in range(m):
                if 0 < alpha[i1] < C:  ## loop the non-bound alpha
                    num_changed += examine_example(i1)
        if examine_all:
            examine_all = 0
        elif num_changed = 0:
            examine_all = 1

def examine_example(i2):
    a2 = alpha[i2]; y2 = y[i2]
    E2 = f(x[i2]) - y2  # f is the svm decision function
    E_cache[i2] = E2
    r2 = E2 * y2
    if (r2 < -tol and a2 < C) or (r2 > tol and a2 > 0):
        if len( non_bound_alpha ) > 1:
            i1 = argmax(E_cache[i1] - E_cache[i2])
            if take_step(i1, i2):
                return 1
        for i1 in non_bound_alpha starting at random:
            if take_step(i1, i2):
                return 1
        for i1 in all_alpha starting at random:
            if take_step(i1, i2):
                return 1
    return 0

def take_step(i1, i2):
    if i1 == i2: return 0
    a1 = alpha[i1]; a2 = alpha[i2]; y1 = y[i1]; y2 = y[i2]
    E1 = f(x[i1]) - y1; E_cache[i1] = E1
    s = y1 * y2
    if y1 == y2:
        L = max(0, a2+a1-C); H = min(C, a2+a1)
    else:
        L = max(0, a2-a1); H = min(C, C+a2-a1)
    if L == H: return 0
    k11 = kernel(x[i1], x[i2]); k12 = kernel(x[i1], x[i2]); k22 = kernel(x[i2], x[i2])
    eta = k11 + k22 - 2 * k12
    if eta > 0:
        a2_new_unclipped = a2 + y2 * (E1 - E_cache[i2]) / eta
        a2_new = clip(a2_new_unclipped, L, H)
    else:
        f1 = y1*(E1+b) - a1*K11 - s*a2*k12; f2 = y2*(E2+b) - s*a1*k12 - a2*k22
        L1 = a1 + s*(a2 - L); H1 = a1 + s*(a2 - H)
        L_obj = L1*f1 + L*f2 + L1**2*k11/2 + L**2*k22/2 + s*L*L1*k12
        H_obj = H1*f1 + H*f2 + H1**2*k11/2 + H**2*k22/2 + s*H*H1*k12
        if L_obj < H_obj - eps: a2_new = L
        elif L_obj > H_obj + eps: a2_new = H
        else: a2_new = a2

    if abs(a2_new - a2) < eps * (a2_new + a2 + eps):
        return 0
    a1_new = a1 + s * (a2 - a2_new)
    # note that in the Platt paper it uses wx-b, which is not common now
    # here we use wx+b, so the equation is different
    b1 = -E1 - y1*(a1_new - a1)*k11 - y2*(a2_new - a2)*k12 + b
    b2 = -E2 - y1*(a1_new - a1)*k12 - y2*(a2_new - a2)*k22 + b
    b = (b1 + b2) / 2
    E_cache[i1] = f(x[i1]) - y1; E_cache[i2] = f(x[i2]) - y2
    return 1
```

