"""RF-greedy with plain scikit-learn defaults, refit after every batch (cluster-blind).

seed set = default_rng(seed).permutation(n)[:S]; fit RandomForestClassifier() (100 trees, sqrt features, msl 1, no class weights) on all labelled peptides; score every unlabelled
peptide; query the top B = 50 (exact ties broken at random); repeat until max_queries. A pickle checkpoint every 100 rounds makes a run resumable bit for bit.
run() returns order, fit_s, score_s, top_p, hits."""
import pickle, time
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def save_ckpt(path, rng, order, fit_s, score_s, top_p, hits, t):
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'wb') as fh: pickle.dump(dict(rng=rng.bit_generator.state, order=order, fit_s=fit_s, score_s=score_s, top_p=top_p, hits=hits, t=t), fh)
    tmp.replace(path)

def run(X, y, seed, S, B=50, max_queries=30000, n_jobs=1, log_every=100, ckpt=None, ckpt_every=100):
    n = len(y); npos = int(y.sum()); rng = np.random.default_rng(seed); perm = rng.permutation(n); seed_idx = perm[:S]
    avail = np.ones(n, dtype=bool); avail[seed_idx] = False; labelled = seed_idx.copy(); order = [seed_idx.astype(np.int32)]
    fit_s, score_s, top_p, hits = [], [], [], []; queried, cum = S, int(y[seed_idx].sum())
    print(f'seed {seed}: N {n}, positives {npos}, seed set {S}, positives in seed {cum}', flush=True)
    t, p = 0, None
    if ckpt is not None and Path(ckpt).exists():
        c = pickle.load(open(ckpt, 'rb')); rng.bit_generator.state = c['rng']
        order, fit_s, score_s, top_p, hits, t = c['order'], c['fit_s'], c['score_s'], c['top_p'], c['hits'], c['t']
        labelled = np.concatenate(order); avail[labelled] = False; queried, cum = len(labelled), int(y[labelled].sum())
        print(f'  resumed from {ckpt} at round {t}, queried {queried}, positives {cum}', flush=True)
    while queried < max_queries and avail.any():
        if ckpt is not None and t > 0 and t % ckpt_every == 0 and len(fit_s) == t: save_ckpt(ckpt, rng, order, fit_s, score_s, top_p, hits, t)
        t0 = time.time()
        rf = RandomForestClassifier(random_state=seed * 100000 + t, n_jobs=n_jobs).fit(X[labelled], y[labelled]); t1 = time.time()
        rows = np.flatnonzero(avail); p = np.full(n, -1.0, dtype=np.float32); p[rows] = rf.predict_proba(X[rows])[:, 1]
        fit_s.append(t1 - t0); score_s.append(time.time() - t1)
        cand = np.flatnonzero(avail); s = p[cand] + 1e-7 * rng.random(len(cand)); b = min(B, len(cand))
        pick = cand[np.argpartition(-s, b - 1)[:b]] if len(cand) > b else cand
        avail[pick] = False; labelled = np.concatenate([labelled, pick]); order.append(pick.astype(np.int32))
        top_p.append(float(np.clip(p[pick], 0, 1).mean())); hits.append(int(y[pick].sum())); queried += len(pick); cum += hits[-1]; t += 1
        if t % log_every == 0:
            print(f'  round {t} queried {queried} positives {cum} ({cum / npos:.1%}) last-{log_every}-round hit rate '
                  f'{np.sum(hits[-log_every:]) / (B * log_every):.3f} fit+score so far {sum(fit_s) + sum(score_s):.0f}s', flush=True)
    return dict(order=np.concatenate(order), fit_s=np.array(fit_s, np.float32), score_s=np.array(score_s, np.float32),
                top_p=np.array(top_p, np.float32), hits=np.array(hits, np.int8))
