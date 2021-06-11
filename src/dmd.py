import numpy as np

class DMD:
    def __init__(self, dt, projected = True, svd_rank = None):
        self.projected = projected
        self.dt = dt
        self.eigenvalues = None
        self.eigenvectors = None
        self.modes = None

    def fit(self, x, rank):
        x1 = x[:, :-1]
        x2 = x[:, 1:]
        u, s, v = np.linalg.svd(x1, full_matrices=False)
        v = v.T

        u = u[:, :rank]
        v = v[:, :rank]
        s = s[:rank]

        a_tilde = np.linalg.multi_dot([u.T, x2, v]) * np.reciprocal(s)
        eigenvalues, eigenvectors = np.linalg.eig(a_tilde)

        if self.projected:
            v11 = u.dot(W)
        else:
            v11 = ((Y.dot(V) *np.reciprocal(Sigma)).dot(W))



    def compute_amplitudes(self):












    