import numpy as np
import matplotlib.pyplot as plt
import os
from src.multical.models.pls import PLS
from src.multical.utils import zscore_matlab_style

def evaluate_and_plot_outliers(x0, absor0_pre, analyte_names, best_k_list, output_dir, colors, conf_level=0.95):
    """
    Evaluates Hotellings T2 and Q-residuals for each analyte based on its 
    selected optimal k latent variables.
    Generates a subplotted Influence plot.
    """
    print("\n--- Evaluating Outliers Post-Calibration ---")
    
    # Time column is dropped during data loading, so we use sample indices
    SampleIdx = np.arange(1, x0.shape[0] + 1)
    X = absor0_pre[1:, :]
    Xnorm, Xmed, Xsig = zscore_matlab_style(X)
    
    nc = len(analyte_names)
    fig, axes = plt.subplots(nc, 1, figsize=(8, 5 * nc))
    if nc == 1:
        axes = [axes]
        
    pls = PLS()
    
    for i, name in enumerate(analyte_names):
        ax = axes[i]
        k_latent = best_k_list[i]
        
        # PLS1 for single analyte
        Y = x0[:, i].reshape(-1, 1)
        Ynorm, Ymed, Ysig = zscore_matlab_style(Y)
        
        try:
            B, T, P, U_scores, Q_loadings, W, r2X, r2Y = pls.nipals(Xnorm, Ynorm, k_latent)
            T2, t2_limit, Q_res, q_limit = pls.compute_outlier_metrics(Xnorm, T, P, conf_level=conf_level)
        except Exception as e:
            print(f"Failed to compute outlier metrics for {name}: {e}")
            continue

        outliers_mask = (T2 > t2_limit) | (Q_res > q_limit)
        outlier_indices = np.where(outliers_mask)[0]
        normal_mask = ~outliers_mask
        
        print(f"[{name}] (LV={k_latent}) T^2 Limit: {t2_limit:.2f} | Q Limit: {q_limit:.2f} | Outliers: {len(outlier_indices)}")
        if len(outlier_indices) > 0:
             print(f"  -> Outlier Indices: {SampleIdx[outliers_mask]}")
        
        c = colors[i] if i < len(colors) else 'blue'
        
        ax.scatter(T2[normal_mask], Q_res[normal_mask], c=c, alpha=0.6, label='Normal')
        
        if len(outlier_indices) > 0:
            ax.scatter(T2[outliers_mask], Q_res[outliers_mask], c='k', alpha=0.8, marker='x', label='Outlier')
            # Annotate outliers with identifier
            for idx in outlier_indices:
                ax.annotate(f"{SampleIdx[idx]}", (T2[idx], Q_res[idx]), 
                            xytext=(5, 5), textcoords='offset points', fontsize=9)
                
        ax.axvline(t2_limit, color='k', linestyle='--', label=f'$T^2$ Limit ({conf_level*100:.0f}%)')
        ax.axhline(q_limit, color='k', linestyle=':', label=f'$Q$ Limit ({conf_level*100:.0f}%)')
        
        letter = chr(97 + i)
        ax.set_xlabel("Hotelling's $T^2$")
        ax.set_ylabel("$Q$-residuals (SPE)")
        ax.set_title(f"{letter}) Influence Plot: {name} (LVs={k_latent})", loc='left')
        ax.legend(loc='best')
        
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "Influence_Plot.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved Influence Plot to {out_path}")
    
    # We DO NOT close the plot, relying on plt.show() in the main runner to show it!
