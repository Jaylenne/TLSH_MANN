import numpy as np
from scipy import sparse
import scipy.sparse.linalg


class simArrayPy:
    def __init__(self, g0, rw=1.0):
        # self.g0
        self.g_true = g0
        self.rw = rw

        self.geff = self.calc_geff(g0, rw)

    def calc_geff(self, g_true, rw: float = 0):
        M = g_true.shape[0]
        N = g_true.shape[1]
        Gw = 1/rw

        # Coordinates
        a = np.arange(M*N).reshape(M, N)
        b = M*N + np.arange(M*N).reshape(N, M).T

        # Calculate sparse matrix A
        xs = np.zeros((M, N, 7))
        ys = np.zeros((M, N, 7))

        xs[:, :, 0] = a
        ys[:, :, 0] = a

        xs[:, :, 1] = b
        ys[:, :, 1] = b

        xs[:, :, 2] = a
        xs[:, -1, 2] = -1
        ys[:, :, 2] = a+1

        xs[:, :, 3] = a
        xs[:, 0, 3] = -1
        ys[:, :, 3] = a-1

        xs[:, :, 4] = b
        xs[-1, :, 4] = -1
        ys[:, :, 4] = b+1

        xs[:, :, 5] = b
        xs[0, :, 5] = -1
        ys[:, :, 5] = b-1

        xs[:, :, 6] = b
        ys[:, :, 6] = a

        v0 = np.zeros((M, N, 7))

        v0[:, :, 0:2] = 2*Gw
        v0[:, :, 2:6] = -Gw
        v0[:, :, -1] = 0

        v0[:, -1, 0] -= Gw
        v0[0, :, 1] -= Gw

        vals = v0
        vals[:, :, 0] += g_true
        vals[:, :, 1] += g_true
        vals[:, :, -1] -= g_true

        m = xs >= 0
        A = sparse.coo_matrix(
            (vals[m], (xs[m], ys[m])), shape=(2*M*N, 2*M*N))

        # Calculate matrix B
        B = np.zeros((2*M*N, M))
        idx = np.ravel_multi_index(
            [np.arange(M)*N, np.arange(M)], dims=B.shape)
        B.ravel()[idx] = -Gw

        # Solve the equation, calculate V top-to bottom, for identify matrix input
        # Vtb, _, _, _ = np.linalg.lstsq(-A.toarray(), B, rcond=None)
        # Sparse matrix version
        ## This is the most computation expensive step in the simulation
        Vtb = sparse.linalg.spsolve(-A.tocsr(), B)

        # Voltage drop between each nodes
        Vd = Vtb[a.T, :] - Vtb[b.T, :]
        Vd = Vd.reshape(-1, Vd.shape[-1]).T

        # Current flows through each device
        Id = Vd * g_true.T.ravel()
        Geff = Id.reshape(M, N, M).sum(axis=2)

        return Geff

    def read_current(self, v_in):
        assert v_in.shape[0] == self.geff.shape[0]

        return self.geff.T @ v_in

    def read_conductance(self):
        pass
