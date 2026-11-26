"""Arm-selection policies.

A policy sees the current Beta posteriors of the *active* (non-exhausted) arms
and returns one score per active arm. The simulator takes the top-k arms by
score. Policies that need a different selection rule (epsilon-greedy)
override `choose`.
"""
import numpy as np


class State:
    """Mutable view handed to policies each round."""

    __slots__ = ("alpha", "beta", "act", "rng", "t")

    def __init__(self, alpha, beta, act, rng, t):
        self.alpha = alpha  # posterior alpha for the active arms
        self.beta = beta    # posterior beta for the active arms
        self.act = act      # indices of the active arms
        self.rng = rng
        self.t = t          # round index (0-based)


class Policy:
    name = "policy"

    def scores(self, st):
        raise NotImplementedError

    def choose(self, scores, m, st):
        """Positions (into `scores`) of the m arms to query, best first."""
        if m >= len(scores):
            return np.argsort(-scores)
        top = np.argpartition(-scores, m - 1)[:m]
        return top[np.argsort(-scores[top])]


def _mean(st):
    return st.alpha / (st.alpha + st.beta)


def _jitter(st):
    # break exact ties at random instead of by arm index
    return st.rng.random(len(st.alpha)) * 1e-9


class Thompson(Policy):
    name = "ts"

    def scores(self, st):
        return st.rng.beta(st.alpha, st.beta)


class Greedy(Policy):
    name = "greedy"

    def scores(self, st):
        return _mean(st) + _jitter(st)


class UCB(Policy):
    """posterior mean + c * posterior SD (Beta closed form)."""

    name = "ucb"

    def __init__(self, c=1.0):
        self.c = float(c)

    def scores(self, st):
        s = st.alpha + st.beta
        sd = np.sqrt(st.alpha * st.beta / (s * s * (s + 1.0)))
        return _mean(st) + self.c * sd + _jitter(st)


class EpsGreedy(Policy):
    """Each of the m slots explores a random arm with prob. eps_t, else takes
    the next-best arm by posterior mean. eps_t = eps * exp(-t / tau) if tau."""

    name = "eps_greedy"

    def __init__(self, eps=0.1, tau=None):
        self.eps = float(eps)
        self.tau = None if tau in (None, 0) else float(tau)

    def scores(self, st):
        return _mean(st) + _jitter(st)

    def choose(self, scores, m, st):
        eps = self.eps if self.tau is None else self.eps * np.exp(-st.t / self.tau)
        n_explore = int(st.rng.binomial(m, eps)) if m < len(scores) else 0
        order = np.argsort(-scores)
        if n_explore == 0:
            return order[:m]
        top = order[: m - n_explore]
        explore = st.rng.choice(order[m - n_explore:], size=n_explore, replace=False)
        return np.concatenate([top, explore])


class ClusterUniform(Policy):
    """No learning: every non-exhausted arm is equally likely to be chosen."""

    name = "cluster_uniform"

    def scores(self, st):
        return st.rng.random(len(st.alpha))


POLICIES = {
    "ts": Thompson,
    "greedy": Greedy,
    "ucb": UCB,
    "eps_greedy": EpsGreedy,
    "cluster_uniform": ClusterUniform,
}


def make_policy(name, **kw):
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}")
    cls = POLICIES[name]
    import inspect

    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    return cls(**{k: v for k, v in kw.items() if k in accepted})
