"""
Test MIWAE + not-MIWAE (base + extended) avec 3 architectures pour le missing model
Self-censoring + pairwise column dependencies
"""

#Need to run it with python 3.7

import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.getcwd())
from MIWAE import MIWAE
from notMIWAE import notMIWAE
from notMIWAE_extended_clean import notMIWAE_extended
import trainer
import utils
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


#Introduction of the missingness process
def introduce_structured_missing(X):
    N, D = X.shape
    Xnan = X.copy()
    half_D = D // 2
    mean_first_half = np.mean(Xnan[:, :half_D], axis=0)
    Xnan[:, :half_D][Xnan[:, :half_D] > mean_first_half] = np.nan
    for i in range(half_D):
        missing_in_first = np.isnan(Xnan[:, i])
        if i < D - half_D:
            Xnan[:, half_D + i][missing_in_first] = np.nan
    Xz = Xnan.copy()
    Xz[np.isnan(Xnan)] = 0
    return Xnan, Xz

# ---- data & training settings
name = '/tmp/uci/task_extended/test'
n_hidden = 128
n_latent_u = 10
n_samples = 20
max_iter = 10000
batch_size = 16
L = 5000

missing_process_base = 'nonlinear'
missing_architectures = ['concat', 'additive', 'multiplicative']
runs = 1

results = {
    'MIWAE': [],
    'notMIWAE_base': [],
}
for arch in missing_architectures:
    results[f'notMIWAE_ext_{arch}'] = []
results.update({'Mean': [], 'MICE': [], 'missForest': []})

for run_idx in range(runs):
    print(f"\n{'='*60}\nRUN {run_idx + 1}/{runs}\n{'='*60}\n")

    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    data = np.array(pd.read_csv(url, sep=';', low_memory=False))[:, :-1]
    N, D = data.shape
    dl = D - 1
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    data = data[np.random.permutation(N)]
    Xtrain = data.copy()
    Xval_org = data.copy()

    Xnan, Xz = introduce_structured_missing(Xtrain)
    S = (~np.isnan(Xnan)).astype(float)
    Xval, Xvalz = introduce_structured_missing(Xval_org)
    missing_rate = 1 - np.mean(S)
    print(f"Missing rate: {missing_rate*100:.2f}%")

    # ---- MIWAE
    print("\nTraining MIWAE...")
    miwae = MIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, name=name)
    trainer.train(miwae, batch_size=batch_size, max_iter=max_iter, name=name + '_miwae')
    rmse_miwae = utils.imputationRMSE(miwae, Xtrain, Xz, Xnan, S, L)[0]
    results['MIWAE'].append(rmse_miwae)
    print(f"MIWAE RMSE: {rmse_miwae:.5f}")

    # ---- not-MIWAE base
    print("\nTraining not-MIWAE (base)...")
    notmiwae = notMIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden,
                        missing_process=missing_process_base, name=name)
    trainer.train(notmiwae, batch_size=batch_size, max_iter=max_iter, name=name + '_notmiwae')
    rmse_notmiwae = utils.not_imputationRMSE(notmiwae, Xtrain, Xz, Xnan, S, L)[0]
    results['notMIWAE_base'].append(rmse_notmiwae)
    print(f"not-MIWAE (base) RMSE: {rmse_notmiwae:.5f}")

    # ---- not-MIWAE extended: loop over architectures
    for arch in missing_architectures:
        print(f"\nTraining not-MIWAE EXTENDED ({arch})...")
        notmiwae_ext = notMIWAE_extended(Xnan, Xval,
                                         n_latent=dl,
                                         n_latent_u=n_latent_u,
                                         n_samples=n_samples,
                                         n_hidden=n_hidden,
                                         missing_process='nonlinear',
                                         missing_model_architecture=arch,
                                         name=name)
        trainer.train(notmiwae_ext, batch_size=batch_size, max_iter=max_iter,
                     name=name + f'_notmiwae_ext_{arch}')
        rmse_ext = utils.not_imputationRMSE(notmiwae_ext, Xtrain, Xz, Xnan, S, L)[0]
        results[f'notMIWAE_ext_{arch}'].append(rmse_ext)
        print(f"not-MIWAE EXTENDED ({arch}) RMSE: {rmse_ext:.5f}")

    # ---- Baselines
    # Mean
    Xrec = SimpleImputer(strategy='mean').fit_transform(Xnan)
    rmse_mean = np.sqrt(np.sum((Xtrain - Xrec)**2 * (1 - S)) / np.sum(1 - S))
    results['Mean'].append(rmse_mean)
    # MICE
    Xrec = IterativeImputer(max_iter=10, random_state=0).fit_transform(Xnan)
    rmse_mice = np.sqrt(np.sum((Xtrain - Xrec)**2 * (1 - S)) / np.sum(1 - S))
    results['MICE'].append(rmse_mice)
    # missForest
    Xrec = IterativeImputer(estimator=RandomForestRegressor(n_estimators=100)).fit_transform(Xnan)
    rmse_rf = np.sqrt(np.sum((Xtrain - Xrec)**2 * (1 - S)) / np.sum(1 - S))
    results['missForest'].append(rmse_rf)

# ---- summary
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
for method, rmses in results.items():
    print(f"{method:25s} = {np.mean(rmses):.5f} ± {np.std(rmses):.5f}")

# Save results
os.makedirs('/tmp/uci/task_extended/', exist_ok=True)
import pickle
with open('/tmp/uci/task_extended/results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("\nResults saved to /tmp/uci/task_extended/results.pkl")
