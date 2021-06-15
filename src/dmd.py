import numpy as np

class DMD:
    def __init__(self, projected=False):
        self.projected = projected
        self.snapshots = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.omegas = None
        self.modes = None
        self.coefficients = None
        self.dt = None
        self.dim = None

    def solve(self, x, dt, rank):
        """
        :param x:
        :param dt:
        :param rank:
        :return:
        """
        self.clear()
        self.dim = x.shape[1:]
        x = DMD.create_tallskinny(x)
        self.snapshots = x
        self.dt = dt
        x1 = x[:, :-1]
        x2 = x[:, 1:]

        u, s, v = np.linalg.svd(x1, full_matrices=False)
        v = v.T
        u = u[:, :rank]
        v = v[:, :rank]
        s = s[:rank]

        a_tilde = np.linalg.multi_dot([u.T, x2, v]) * np.reciprocal(s)
        eigenvalues, eigenvectors = np.linalg.eig(a_tilde)

        self.eigenvalues = eigenvalues
        self.omegas = np.log(self.eigenvalues)/self.dt
        self.eigenvectors = eigenvectors
        if self.projected:
            self.modes = u.dot(eigenvectors)
        else:
            self.modes = ((x2.dot(v) * np.reciprocal(s)).dot(eigenvectors))

        self.compute_coefficients(False)

    def compute_coefficients(self, exact = False):
        """

        :param exact:
        :return:
        """
        if exact:
            self.coefficients = np.linalg.solve(self.modes, self.snapshots[:, 0])
        else:
            self.coefficients = np.linalg.lstsq(self.modes, self.snapshots[:, 0])[0]

    def evolve(self, time_array=None, mask=None):
        """
        Evolves the dynamical system in time according to the
        eigendecomposition of A.
        :param time_array: array of time for which to approximate state.
            If None, then approximate state at the time points
            where snapshots are taken
        :return: array where each column is approximated state at time t
        """

        masked_eigenvalues = self.eigenvalues
        masked_modes = self.modes
        if mask is not None:
            masked_eigenvalues[~mask] = 0
            masked_modes[:, ~mask] = 0

        if time_array is None:
            num_snapshots = self.snapshots.shape[1]
            tbase = np.outer(masked_eigenvalues, np.ones(num_snapshots))
            tpower = np.arange(start=0, stop=num_snapshots)
        else:
            tbase = np.outer(masked_eigenvalues, np.ones(time_array.size))
            tpow = time_array/self.dt

        y1 = np.power(tbase, tpower)
        y2 = y1 * self.coefficients[:, None]
        return masked_modes.dot(y2)

    def create_mask(self, threshold = 0.1):
        return np.abs(self.omegas) < threshold

    def clear(self):
        self.snapshots = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.omegas = None
        self.modes = None
        self.coefficients = None
        self.dt = None
        self.dim = None

    def plot_omegas(self, ax):
        return ax.scatter(np.real(self.omegas), np.imag(self.omegas))


    @staticmethod
    def create_tallskinny(x):
        """
        :param x:
        :return:
        """
        cols = x.shape[0]
        rows = int(x.size/x.shape[0])
        data_matrix = np.zeros(shape=(rows, cols))

        for i in range(cols):
            data_matrix[:, i] = x[i].flatten(order = 'C')

        return data_matrix
