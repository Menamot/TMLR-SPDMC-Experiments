import numpy as np
from itertools import combinations

from dbc.utils import compute_prior, proj_simplex_Condat


def proj_onto_U(pi, Box, K):
    """
    Parameters
    ----------
    pi : Array of floats
        Vector to project onto the box-constrained simplex..
    Box : Matrix
        {'none', matrix} : Box-constraint on the priors.
    K : int
        Number of classes.

    Returns
    -------
    pi_new : Array of floats
            Priors projected onto the box-constrained simplex.

    """

    def proj_onto_polyhedral_set(pi, Box, K):
        """
        Parameters
        ----------
        pi : Array of floats
            Vector to project onto the box-constrained simplex.
        Box : Array
            {'none', matrix} : Box-constraint on the priors.
        K : int
            Number of classes.

        Returns
        -------
        piStar : Array of floats
                Priors projected onto the box-constrained simplex.

        """

        def num2cell(a):
            if type(a) is np.ndarray:
                return [num2cell(x) for x in a]
            else:
                return a

        # Verification of constraints
        for i in range(K):
            for j in range(2):
                if Box[i, j] < 0:
                    Box[i, j] = 0
                if Box[i, j] > 1:
                    Box[i, j] = 1

        # Generate matrix G:
        U = np.concatenate((np.eye(K), -np.eye(K), np.ones((1, K)), -np.ones((1, K))))
        eta = Box[:, 1].tolist() + (-Box[:, 0]).tolist() + [1] + [-1]

        n = U.shape[0]

        G = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                G[i, j] = np.vdot(U[i, :], U[j, :])

        # Generate subsets of {1,...,n}:
        M = (2**n) - 1
        I = num2cell(np.zeros((1, M)))

        i = 0
        for l in range(n):
            T = list(combinations(list(range(n)), l + 1))
            for p in range(i, i + len(T)):
                I[0][p] = T[p - i]
            i = i + len(T)

        # Algorithm

        for m in range(M):
            Im = I[0][m]

            Gmm = np.zeros((len(Im), len(Im)))
            ligne = 0
            for i in Im:
                colonne = 0
                for j in Im:
                    Gmm[ligne, colonne] = G[i, j]
                    colonne += 1
                ligne += 1

            if np.linalg.det(Gmm) != 0:

                nu = np.zeros((2 * K + 2, 1))
                w = np.zeros((len(Im), 1))
                for i in range(len(Im)):
                    w[i] = np.vdot(pi, U[Im[i], :]) - eta[Im[i]]

                S = np.linalg.solve(Gmm, w)

                for e in range(len(S)):
                    nu[Im[e]] = S[e]

                if not np.any(nu < -(10 ** (-10))):
                    A = G.dot(nu)
                    z = np.zeros((1, 2 * K + 2))
                    for j in range(2 * K + 2):
                        z[0][j] = np.vdot(pi, U[j, :]) - eta[j] - A[j]

                    if np.all(z <= 10 ** (-10)):
                        pi_new = pi
                        for i in range(2 * K + 2):
                            pi_new = pi_new - nu[i] * U[i, :]

        piStar = pi_new

        # Remove noisy small calculus errors:
        piStar = piStar / piStar.sum()

        return piStar

    check_U = 0
    if pi.sum() == 1:
        for k in range(K):
            if (pi[0][k] >= Box[k, 0]) & (pi[0][k] <= Box[k, 1]):
                check_U = check_U + 1

    if check_U == K:
        pi_new = pi

    if check_U < K:
        pi_new = proj_onto_polyhedral_set(pi, Box, K)

    return pi_new


def compute_piStar(pHat, y_train, K, L, N, Box):
    """
    Parameters
    ----------
    pHat : Array of floats
        Probability estimate of observing the features profile in each class.
    y_train : Dataframe
        Real labels of the training set.
    K : int
        Number of classes.
    L : Array
        Loss Function.
    N : int
        Number of iterations in the projected subgradient algorithm.
    Box : Array
        {'none', matrix} : Box-constraints on the priors.

    Returns
    -------
    piStar : Array of floats
        Least favorable priors.
    rStar : float
        Global risks.
    RStar : Array of float
        Conditional risks.
    V_iter : Array
        Values of the V function at each iteration.
    stockpi : Array
        Values of pi at each iteration.

    """

    # IF BOX-CONSTRAINT == NONE (PROJECTION ONTO THE SIMPLEX)
    def compute_global_risk(conditional_risk, prior):
        global_risk = np.sum(conditional_risk * prior)
        return global_risk

    if Box is None:
        pi = compute_prior(y_train).reshape(1, -1)
        rStar = 0
        piStar = pi
        RStar = 0

        V_iter = []
        stockpi = np.zeros((K, N))

        for n in range(1, N + 1):
            # Compute subgradient R at point pi (see equation (21) in the paper)
            lambd = np.dot(L, pi.T * pHat)
            R = np.zeros((1, K))

            mu_k = np.sum(L[:, np.argmin(lambd, axis=0)] * pHat, axis=1)
            R[0, :] = mu_k
            stockpi[:, n - 1] = pi[0, :]

            r = compute_global_risk(R, pi)
            V_iter.append(r)
            if r > rStar:
                rStar = r
                piStar = pi
                RStar = R
                # Update pi for iteration n+1
            gamma = 1 / n
            eta = np.maximum(float(1), np.linalg.norm(R))
            w = pi + (gamma / eta) * R
            pi = proj_simplex_Condat(K, w)

        # Check if pi_N == piStar
        lambd = np.dot(L, pi.T * pHat)
        R = np.zeros((1, K))

        mu_k = np.sum(L[:, np.argmin(lambd, axis=0)] * pHat, axis=1)
        R[0, :] = mu_k
        stockpi[:, n - 1] = pi[0, :]

        r = compute_global_risk(R, pi)
        if r > rStar:
            rStar = r
            piStar = pi
            RStar = R

    # IF BOX-CONSTRAINT
    else:
        pi = compute_prior(y_train).reshape(1, -1)
        rStar = 0
        piStar = pi
        RStar = 0

        V_iter = []
        stockpi = np.zeros((K, N))

        for n in range(1, N + 1):
            # Compute subgradient R at point pi (see equation (21) in the paper)
            lambd = np.dot(L, pi.T * pHat)
            R = np.zeros((1, K))

            mu_k = np.sum(L[:, np.argmin(lambd, axis=0)] * pHat, axis=1)
            R[0, :] = mu_k
            stockpi[:, n - 1] = pi[0, :]

            r = compute_global_risk(R, pi)
            V_iter.append(r)
            if r > rStar:
                rStar = r
                piStar = pi
                RStar = R
                # Update pi for iteration n+1
            gamma = 1 / n
            eta = np.maximum(float(1), np.linalg.norm(R))
            w = pi + (gamma / eta) * R
            pi = proj_onto_U(w, Box, K)

        # Check if pi_N == piStar
        lambd = np.dot(L, pi.T * pHat)
        R = np.zeros((1, K))

        mu_k = np.sum(L[:, np.argmin(lambd, axis=0)] * pHat, axis=1)
        R[0, :] = mu_k
        stockpi[:, n - 1] = pi[0, :]

        r = compute_global_risk(R, pi)
        if r > rStar:
            rStar = r
            piStar = pi
            RStar = R

    return piStar, rStar, RStar, V_iter, stockpi
