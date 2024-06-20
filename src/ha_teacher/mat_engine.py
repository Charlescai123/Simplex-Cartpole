import io
import copy
import matlab
import matlab.engine
import numpy as np
import cvxpy as cp
from omegaconf import DictConfig


class MatEngine:

    def __init__(self, cfg: DictConfig):
        self.engine = None

        # Matlab engine output
        self.out = io.StringIO() if cfg.stdout is False else None
        self.err = io.StringIO() if cfg.stderr is False else None

        self.matlab_engine_launch(working_path=cfg.working_path)

        # CVX Tool Setup
        if cfg.cvx_toolbox.setup:
            self.cvx_setup(cvx_path=cfg.cvx_toolbox.relative_path)

    def matlab_engine_launch(self, working_path="src/ha_teacher/matlab/"):
        print("Launching Matlab Engine...")
        self.engine = matlab.engine.start_matlab()
        self.engine.cd(working_path)
        print("Matlab current working directory is ---->>>", self.engine.pwd())

    def cvx_setup(self, cvx_path="./cvx"):
        current_path = self.engine.pwd()
        self.engine.cd(cvx_path)
        print("Setting up the CVX Toolbox...")
        _ = self.engine.cvx_setup
        print("CVX Toolbox setup done, switch back to original working path")
        self.engine.cd(current_path)

    def system_patch(self,
                     As: np.ndarray,
                     Bs: np.ndarray,
                     Ak: np.ndarray,
                     Bk: np.ndarray):
        As = As.reshape(4, 4)
        Bs = Bs.reshape(4, 1)
        Ak = Ak.reshape(4, 4)
        Bk = Bk.reshape(4, 1)

        As = matlab.double(As.tolist())
        Bs = matlab.double(Bs.tolist())
        Ak = matlab.double(Ak.tolist())
        Bk = matlab.double(Bk.tolist())

        F_hat, t_min = self.engine.patch_lmi(As, Bs, Ak, Bk, nargout=2, stdout=self.out, stderr=self.err)

        return F_hat, t_min

    def feedback_control_cvxpy(self, Ac, Bc, Ak, Bk, sc, sd):
        Ac = Ac.reshape(4, 4)
        Bc = Bc.reshape(4, 1)
        Ak = Ak.reshape(4, 4)
        Bk = Bk.reshape(4, 1)
        sc = sc.reshape(4, 1)
        sd = sd.reshape(4, 1)

        # Constants
        n = 4
        alpha = 0.96

        # Calculating error and its absolute value
        e = sc - sd
        val = np.abs(sc - sd)

        # Define D matrix
        D = np.array([[1 / 0.4, 0, 0, 0],
                      [0, 1 / 4.5, 0, 0],
                      [0, 0, 1 / 0.4, 0],
                      [0, 0, 0, 1 / 4.5]])

        # Define CVXPY variables
        Q = cp.Variable((n, n), symmetric=True)
        R = cp.Variable((1, n))

        # Define CVXPY optimization problem
        objective = cp.Maximize(cp.log_det(Q))
        print(alpha * Q, Q @ Ak.T + R.T @ Bk.T)
        constraints = [cp.bmat([[alpha * Q, Q @ Ak.T + R.T @ Bk.T],
                                [Ak @ Q + Bk @ R, Q]]) >> 0,
                       D @ Q @ D.T - np.eye(4) << 0]

        prob = cp.Problem(objective=objective, constraints=constraints)

        # Solve the problem
        prob.solve()

        # Extract solution
        K = np.array(R.value) @ np.linalg.pinv(Q.value)
        M = Ac + Bc @ K

        # Check stability
        assert np.all(np.linalg.eigvals(M) < 0)

        return K


if __name__ == '__main__':
    As = np.array([[0, 1, 0, 0],
                   [0, 0, -1.42281786576776, 0.182898194776782],
                   [0, 0, 0, 1],
                   [0, 0, 25.1798795199119, 0.385056459685276]])

    Bs = np.array([[0,
                    0.970107410065162,
                    0,
                    -2.04237185222105]])

    Ak = np.array([[1, 0.0100000000000000, 0, 0],
                   [0, 1, -0.0142281786576776, 0.00182898194776782],
                   [0, 0, 1, 0.0100000000000000],
                   [0, 0, 0.251798795199119, 0.996149435403147]])

    Bk = np.array([[0,
                    0.00970107410065163,
                    0,
                    -0.0204237185222105]])

    sd = np.array([[0.234343490000000,
                    0,
                    -0.226448960000000,
                    0]])

    mat = MatEngine()
    K = mat.feedback_law(As, Bs, Ak, Bk, sd)
    print(K)
