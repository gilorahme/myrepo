"""Step 32: Diagnostic analysis only; does not fit or optimize a trading model.
Run: python regime_analysis.py
Requires: pandas, numpy, matplotlib
Inputs: data/BTC_ML_features.csv, results/final_holdout_2026.csv
Optional: results/ml_trades_75.csv (development trade comparison)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURES = [
    'return_1h','return_6h','return_12h','return_24h','return_72h',
    'price_vs_ema20','price_vs_ema50','price_vs_ema200','ema50_vs_ema200',
    'rsi_14','volatility_24h','volatility_72h','range_pct','volume_ratio'
]
BASE = Path('.')
OUT = BASE / 'results' / 'regime_diagnostics'
OUT.mkdir(parents=True, exist_ok=True)

def load(path):
    df = pd.read_csv(path)
    for c in ('timestamp','signal_time','entry_time','exit_time'):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors='raise')
    return df

def psi(reference, current, bins=10):
    a = pd.to_numeric(reference, errors='coerce').replace([np.inf,-np.inf],np.nan).dropna().to_numpy()
    b = pd.to_numeric(current, errors='coerce').replace([np.inf,-np.inf],np.nan).dropna().to_numpy()
    if len(a)<bins or len(b)<bins:
        return np.nan
    edges = np.unique(np.quantile(a, np.linspace(0,1,bins+1)))
    if len(edges)<3:
        return np.nan
    edges[0],edges[-1] = -np.inf,np.inf
    pa = np.histogram(a,bins=edges)[0]/len(a)
    pb = np.histogram(b,bins=edges)[0]/len(b)
    pa,pb = np.maximum(pa,1e-6),np.maximum(pb,1e-6)
    return float(np.sum((pb-pa)*np.log(pb/pa)))

def trade_enrichment(trades, features, label):
    if 'signal_time' not in trades or 'net_return' not in trades:
        print(f'{label}: missing signal_time or net_return; skipping')
        return
    left = trades.copy().sort_values('signal_time')
    right = features[['timestamp']+FEATURES+['close']].sort_values('timestamp')
    enriched = pd.merge_asof(left, right, left_on='signal_time',right_on='timestamp', direction='backward', tolerance=pd.Timedelta(hours=1))
    if enriched[FEATURES].isna().any(axis=None):
        print(f'WARNING {label}: {enriched[FEATURES].isna().any(axis=1).sum()} unmatched trade signals')
    enriched['result'] = np.where(enriched['net_return']>0,'winner','loser')
    enriched.to_csv(OUT/f'{label}_enriched.csv',index=False)
    cols=['net_return']+FEATURES
    summary=enriched.groupby('result')[cols].agg(['mean','median','count'])
    summary.to_csv(OUT/f'{label}_winner_loser_summary.csv')
    print(f'\n{label}: {len(enriched)} trades; {sum(enriched.net_return>0)} winners')
    print(enriched.groupby('result')[['net_return','rsi_14','volatility_24h','return_24h','ema50_vs_ema200']].mean().round(5).to_string())
    if 'probability' in enriched:
        print('Confidence vs realized net return (Spearman):', round(enriched[['probability','net_return']].corr(method='spearman').iloc[0,1],3))
    return enriched

def main():
    features=load(BASE/'data'/'BTC_ML_features.csv').sort_values('timestamp').drop_duplicates('timestamp')
    holdout=load(BASE/'results'/'final_holdout_2026.csv')
    missing=[c for c in FEATURES if c not in features]
    if missing: raise ValueError(f'Missing feature columns: {missing}')
    dev=features[features.timestamp < pd.Timestamp('2026-01-01',tz='UTC')]
    test=features[features.timestamp >= pd.Timestamp('2026-01-01',tz='UTC')]
    if dev.empty or test.empty: raise ValueError('Both development and 2026 data are required')
    print('Data:',len(features),'rows;',len(dev),'development;',len(test),'2026')
    print('Range:',features.timestamp.min(),'to',features.timestamp.max())
    print('Duplicate timestamps:',features.timestamp.duplicated().sum())
    gaps=features.timestamp.diff().dropna()
    print('Non-hourly gaps:',int((gaps != pd.Timedelta(hours=1)).sum()))
    if 'future_close_24h' in features:
        expected=features['close'].shift(-24)
        valid=expected.notna() & features['future_close_24h'].notna()
        mismatch=~np.isclose(expected[valid],features.loc[valid,'future_close_24h'],rtol=1e-7,atol=1e-5)
        print('Future-close mismatch count:',int(mismatch.sum()),'of',int(valid.sum()))
        tail=features.tail(24)
        if tail['future_close_24h'].notna().any():
            print('WARNING: final 24 feature rows contain future-close labels; verify source extends beyond feature file end and labels were generated chronologically.')
    rows=[]
    for c in FEATURES:
        rows.append(dict(feature=c,dev_mean=dev[c].mean(),holdout_mean=test[c].mean(),dev_std=dev[c].std(),holdout_std=test[c].std(),psi=psi(dev[c],test[c])))
    drift=pd.DataFrame(rows).sort_values('psi',ascending=False)
    drift.to_csv(OUT/'feature_drift.csv',index=False)
    print('\nFeature drift (PSI):\n',drift[['feature','psi','dev_mean','holdout_mean']].round(5).to_string(index=False))
    annual=[]
    for year,g in features.groupby(features.timestamp.dt.year):
        annual.append(dict(year=year,rows=len(g),first=g.timestamp.iloc[0],last=g.timestamp.iloc[-1],price_change=g.close.iloc[-1]/g.close.iloc[0]-1,avg_rsi=g.rsi_14.mean(),avg_volatility_24h=g.volatility_24h.mean(),avg_return_24h=g.return_24h.mean()))
    pd.DataFrame(annual).to_csv(OUT/'annual_regimes.csv',index=False)
    print('\nAnnual regimes:\n',pd.DataFrame(annual).round(5).to_string(index=False))
    h=trade_enrichment(holdout,features,'holdout_2026')
    dev_path=BASE/'results'/'ml_trades_75.csv'
    if dev_path.exists():
        d=load(dev_path)
        if 'signal_time' in d:
            trade_enrichment(d,features,'development_75')
        else:
            print('\nDevelopment file has no signal_time; cannot reliably join features.')
    else:
        print('\nOptional development trade file not found:',dev_path)
    fig,ax=plt.subplots(figsize=(10,5))
    top=drift.head(14).iloc[::-1]
    ax.barh(top.feature,top.psi)
    ax.set_xlabel('Population Stability Index (PSI)')
    ax.set_title('2026 feature drift vs 2024–2025 development')
    fig.tight_layout()
    fig.savefig(OUT/'feature_drift.png',dpi=160)
    plt.close(fig)
    if h is not None and len(h):
        fig,ax=plt.subplots(figsize=(9,4))
        ax.scatter(h['probability'],h['net_return']*100)
        ax.axhline(0,linestyle='--')
        ax.set(xlabel='Model probability',ylabel='Realized net trade return (%)',title='2026 confidence vs trade return (n=12)')
        fig.tight_layout()
        fig.savefig(OUT/'confidence_vs_return.png',dpi=160)
        plt.close(fig)
    print('\nSaved diagnostics to:',OUT.resolve())
    print('NOTE: PSI is descriptive, not proof of regime causation. Do not use 2026 to retune the frozen strategy.')

if __name__=='__main__': main()
