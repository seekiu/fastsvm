'''This file implements the vanilla SMO algorithm described in the
Platt1998 paper. Here only SVC with non-linear kernel is implemented (linear
kernel has separate optimization available).

The kernel is fully cached.
'''

import time
import warnings

import numpy as np
import numba as nb
from numba.core.errors import NumbaPerformanceWarning



@nb.experimental.jitclass([
    ('C', nb.float64),
    ('gamma', nb.float64),
    ('max_iter', nb.int64),
    ('tol', nb.float64),
    ('_eps', nb.float64),
    ('_info_smo_steps', nb.int64),
    ('_info_obj', nb.float64),
    ('_info_kkt_status', nb.int64[:]),
    ('X', nb.float64[:, :]),
    ('y', nb.float64[:]),
    ('alpha', nb.float64[:]),
    ('b', nb.float64[:]),
    ('c_E', nb.float64[:]),
    ('c_K', nb.float64[:, :]),
])  # yapf: disable
class SVCPlatt:
    def __init__(self,
                 C: float = 1.0,
                 gamma: float = 1.0,
                 max_iter: int = 10000,
                 tol: float = 1e-3) -> None:
        '''SVC that implements the original SMO algorithm from Platt1998 paper
        with sklearn interface.

        Args:
            C: penalty term
            gamma: parameter for RBF kernel
            max_iter: max number of SMO inner loop iterations
            tol: numerical tolerance

        Usage:
            clf = SVCPlatt(C=1.0, gamma=1.0)
            clf.fit(X, y)
            y_hat = clf.predict(X_new)
        '''
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self._info_smo_steps: int = 0
        self._info_kkt_status = np.zeros(5, dtype=np.int64)
        ## there are both tol and eps in the paper, but didn't explain the difference
        ## there is only one eps in libsvm, so here set eps := tol
        self._eps = tol  #0.001  # tolerance for KKT condition check

    def fit(self, X, y) -> None:
        self.X: np.array = X
        self.y: np.array = y
        self.alpha = np.zeros_like(y, dtype=np.float64)
        self.b = np.array([-0.])  ## declare as array for easier inplace update
        self.c_E = np.zeros_like(y, dtype=np.float64)  # cache for E
        self.c_K = -0.01 * np.ones((y.shape[0], y.shape[0]))  # cache for kernels

        ## SMO
        self._build_kernel_cache()
        self._smo_main_loop()
        self._info_kkt_status[:] = self._check_kkt()

    def predict(self, X):
        ## TODO: Add batch prediction
        res = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            res[i] = self._f(X[i])
        return np.sign(res).astype(int)

    def _build_kernel_cache(self):
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                x1 = self.X[i]
                x2 = self.X[j]
                self.c_K[i, j] = kernel_rbf(x1, x2, self.gamma)

    def _smo_main_loop(self):
        alpha, X, y, C, gamma = self.alpha, self.X, self.y, self.C, self.gamma
        num_changed = 0
        examine_all = 1
        while num_changed > 0 or examine_all:
            # if valid_updates[0] > self.max_iter:
            if self._info_smo_steps > self.max_iter:
                print('\nWARNING!!! MAX_ITER REACHED AND NOT CONVERGED!!!\n')
                return
            num_changed = 0
            if examine_all:
                for j in range(X.shape[0]):
                    num_changed += self._examine_example(j)
            else:
                for j in range(X.shape[0]):
                    if 0 < alpha[j] < C:
                        num_changed += self._examine_example(j)
            if examine_all:
                examine_all = 0
            elif num_changed == 0:
                examine_all = 1

        self._info_obj = self._obj_func(alpha, X, y, gamma)

    def _examine_example(self, j: int) -> int:
        alpha, b, X, y, C, gamma, tol, c_K = \
            self.alpha, self.b, self.X, self.y, self.C, self.gamma, self.tol, self.c_K
        c_E = self.c_E
        y2 = y[j]
        a2 = alpha[j]
        # E2 = self._f(X[j]) - y2
        E2 = self._f_from_cache_kernel(j) - y2
        self.c_E[j] = E2
        r2 = E2 * y2

        if (r2 < -tol and a2 < C) or (r2 > tol and a2 > 0):
            if alpha[(alpha > 0) & (alpha < C)].shape[0] > 0:
                if E2 > 0:
                    i = self.c_E.argmin()
                else:
                    i = self.c_E.argmax()
                if self._take_step(i, j):
                    return 1
            indices = np.arange(alpha.shape[0])
            np.random.shuffle(indices)
            for i in indices:
                if 0 < alpha[i] < C:
                    if self._take_step(i, j):
                        return 1

            indices = np.arange(alpha.shape[0])
            np.random.shuffle(indices)
            for i in indices:
                if self._take_step(i, j):
                    return 1

        return 0

    def _take_step(self, i, j):
        """Do one SMO step: Update 2 alphas and b. Updates are inplace.
        """
        if i == j: return 0

        alpha, b, c_E, X, y, C, gamma, tol, c_K = (self.alpha, self.b, self.c_E, self.X, self.y,
                                                   self.C, self.gamma, self.tol, self.c_K)
        self._info_smo_steps += 1
        a1, a2 = alpha[i], alpha[j]
        y1, y2 = y[i], y[j]
        s = y1 * y2
        f1 = self._f_from_cache_kernel(i)
        f2 = self._f_from_cache_kernel(j)
        E1 = f1 - y1
        E2 = f2 - y2
        c_E[i] = E1
        c_E[j] = E2

        if y1 == y2:
            lo = max(0, a1 + a2 - C)
            hi = min(C, a2 + a1)
        else:
            lo = max(0, a2 - a1)
            hi = min(C, C + a2 - a1)

        if hi - lo < 1.e-6: return 0

        k11 = c_K[i, i]  # kernel_rbf(X[i], X[i], gamma)
        k22 = c_K[j, j]  # kernel_rbf(X[j], X[j], gamma)
        k12 = c_K[i, j]  # kernel_rbf(X[i], X[j], gamma)
        eta = k11 + k22 - 2 * k12

        if eta > 0:
            a2_new_unclip = a2 + y2 * (E1 - E2) / (eta + 1e-9)
            a2_new = clip(a2_new_unclip, hi, lo)
        else:  # this happens rarely
            ff1 = y1 * (E1 + b[0]) - a1 * k11 - s * a2 * k12
            ff2 = y2 * (E2 + b[0]) - s * a1 * k12 - a2 * k22
            lo1 = a1 * s * (a2 - lo)
            hi1 = a1 + s * (a2 - hi)
            lo_obj = lo1 * ff1 + lo * ff2 + lo1**2 * k11 / 2 + lo**2 * k22 / 2 + s * lo * lo1 * k12
            hi_obj = hi1 * ff1 + hi * ff2 + hi1**2 * k11 / 2 + hi**2 * k22 / 2 + s * hi * hi1 * k12
            if lo_obj < hi_obj - self._eps:
                a2_new = lo
            elif lo_obj > hi_obj + self._eps:
                a2_new = hi
            else:
                a2_new = a2

        ## Added `C` below, which is not there in the original SMO paper.
        ## Sometimes when a2 is large, the numerical error is unneglible.
        if np.abs(a2_new - a2) < self._eps * (a2_new + a2 + self._eps) / C:
            return 0
        a1_new = a1 + s * (a2 - a2_new)

        ## update b
        b1 = -E1 + y1 * k11 * (a1 - a1_new) + y2 * k12 * (a2 - a2_new) + b[0]
        b2 = -E2 - y1 * k12 * (a1_new - a1) - y2 * k22 * (a2_new - a2) + b[0]
        b[0] = (b1 + b2) / 2.0

        a1_new_norm = a1_new / C
        a2_new_norm = a2_new / C
        if a1_new_norm < 1e-6:
            a1_new = 0
        elif a1_new_norm > 1 - 1e-6:
            a1_new = C
        if a2_new_norm < 1e-6:
            a2_new = 0
        elif a2_new_norm > 1 - 1e-6:
            a2_new = C

        alpha[i] = a1_new
        alpha[j] = a2_new

        ## Doesn't seem necessary to update E cache.
        # f1 = _fast_f(X[i], alpha, b, X, y, gamma)
        # f2 = _fast_f(X[j], alpha, b, X, y, gamma)
        # E1 = f1 - y[i]
        # E2 = f2 - y[j]
        # c_E[i] = E1; c_E[j] = E2

        return 1

    def _f(self, x) -> float:
        X, y, alpha, b, gamma = self.X, self.y, self.alpha, self.b, self.gamma
        res = 0

        for i in range(X.shape[0]):
            res += alpha[i] * y[i] * kernel_rbf(X[i], x, gamma)
        res += b[0]
        return res

    def _f_from_cache_kernel(self, i_x):
        alpha, b, X, y, c_K = self.alpha, self.b, self.X, self.y, self.c_K
        res = 0

        for i in range(X.shape[0]):
            res += alpha[i] * y[i] * c_K[i_x, i]
        res += b[0]
        return res

    def _obj_func(self, alpha, X, y, gamma) -> float:
        res = 0
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                res += alpha[i] * alpha[j] * y[i] * y[j] * kernel_rbf(X[i], X[j], gamma)
        res *= (-0.5)
        res += alpha.sum()
        return res

    def _check_kkt(self):
        '''Check if KKT conditions are satisfied by the given tolerance.
        Currently very computationally intensive. No optimization at all.
        Ref: https://zhuanlan.zhihu.com/p/64580199
        '''
        n_violation = 0
        n_non_sv = 0  # alpha == 0
        n_sv_bound = 0  # 0 < alpha < C
        n_sv_non_bound = 0  # alpha == C
        out_of_bound = 0  # only for sanity check
        tol = 1e-2

        alpha = np.zeros_like(self.alpha)
        alpha[:] = self.alpha[:]
        # np.round(alpha, decimals=9)

        for i in range(self.X.shape[0]):
            E = self._f(self.X[i]) - self.y[i]
            E_yi = E * self.y[i]
            a = alpha[i]
            if -1.e-9 < a < 1.e-9:  # alpha == 0
                n_non_sv += 1
                if E_yi < -tol:
                    n_violation += 1
            elif a < self.C:
                n_sv_non_bound += 1
                if E_yi < -self.tol or E_yi > tol:
                    n_violation += 1
            elif a == self.C:
                n_sv_bound += 1
                if E_yi > tol:
                    n_violation += 1
            else:
                out_of_bound += 1
        # return np.array([n_violation, n_non_sv, n_sv_bound, n_sv_non_bound, out_of_bound])
        return (n_violation, n_non_sv, n_sv_bound, n_sv_non_bound, out_of_bound)


@nb.njit
def clip(x: float, hi: float, lo: float) -> float:
    if x > hi:
        return hi
    if x < lo:
        return lo
    return x


@nb.njit
def l2_norm(x: float) -> float:
    s = 0
    for i in range(x.shape[0]):
        s += (x[i]**2)
    return s


@nb.njit
def kernel_rbf(x1, x2, gamma) -> float:
    ## TODO: currently using gamma=0 to switch to linear kernel
    if gamma < 1e-5:
        return np.dot(x1, x2)
    return np.exp(-gamma * l2_norm(x1 - x2))


def test_correctness():
    # suppress numba's performance warning when calling the np.dot function with
    # non-contiguous arrays
    warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

    test_cases = (
        ## [X, y, alpha_ref, obj_ref, gamma, C]
        (
            np.array([[0, 0], [1, 0], [0, 1], [2, 0], [0, 2.5], [2, 2]], dtype=float),
            np.array([1, 1, 1, -1, -1, -1], dtype=float),
            np.array([0.00000, 3.28000, 0.00000, 2.64000, 0.64000, 0.00000], dtype=float),
            3.2800000097991084,
            0,
            1000
        ),
        (
            np.array([[0, 0], [1, 0], [0, 1], [2, 0], [0, 2.5], [2, 2]], dtype=float),
            np.array([1, 1, 1, -1, -1, -1], dtype=float),
            np.array([0.3669751, 1.4585961, 1.0255837, 1.2813834, 0.8439294, 0.7258422], dtype=float),
            2.851154965169337,
            1.0,
            1000
        ),
        (
            np.array([[0, 0], [1, 0], [0, 1], [1, 1], [3, 0], [0, 3.1], [2, 2]], dtype=float),
            np.array([1, 1, 1, 1, -1, -1, -1], dtype=float),
            np.array([0.8522027, 0.8522024, 0.8522024, 0.8522543, 1.1362744, 1.1362744, 1.1363131], dtype=float),
            3.408861840289703,
            5.0,
            10,
        ),
    ) # yapf: disable

    for case_i, (X, y, alpha_ref, obj_ref, gamma, C) in enumerate(test_cases[:]):
        print('*' * 100)
        clf = SVCPlatt(C=C, max_iter=100_000, gamma=gamma, tol=1e-5)
        clf.fit(X, y)
        print(f'Case {case_i}: smo_steps={clf._info_smo_steps}, KKT={clf._info_kkt_status}.')
        print('Alpha match?', np.allclose(alpha_ref, clf.alpha))
        print('Object match?', np.allclose(clf._info_obj, obj_ref))
        if not np.allclose(clf.alpha, alpha_ref):
            print(f'alpha_ref\t{alpha_ref}\nalpha_mine\t{np.round(clf.alpha, 7)}')


if __name__ == '__main__':
    test_correctness()
