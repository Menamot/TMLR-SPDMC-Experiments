"""Synthetic illustrations used in the paper (run from the repository root)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

OUTPUT = Path("outputs/illustrations")

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    plt.show = lambda *args, **kwargs: None
    with threadpool_limits(limits=1):
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        import matplotlib.pyplot as plt
        
        random_seed = 42
        
        dataset_option = 1
        
        n_class1 = 50 * 5
        n_class2 = 50 * 2
        
        if dataset_option == 1:
            X, y = datasets.make_blobs(
                n_samples=[n_class1, n_class2],
                n_features=2,
                centers=[(9.5, 10), (10, 9.4)],
                cluster_std=[[0.6, 0.6], [0.35, 0.3]],
                shuffle=True,
                random_state=random_seed
            )
        elif dataset_option == 2:
            X, y = datasets.make_circles(
                n_samples=(n_class1, n_class2),
                factor=0.2,
                noise=0.3,
                random_state=random_seed
            )
        elif dataset_option == 3:
            X, y = datasets.make_moons(
                n_samples=(n_class1, n_class2),
                noise=0.15,
                random_state=random_seed
            )
        else : raise ValueError('Invalid dataset option')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
        import dbc
        import numpy as np
        import matplotlib.pyplot as plt
        from dbc.discretizers import CMeansDiscretizer
        from scipy.spatial import Voronoi, voronoi_plot_2d
        
        model_kmeans = dbc.DiscreteBayesianClassifier(
            discretizer=CMeansDiscretizer(n_clusters=9, fuzzifier=1.00000001, use_kmeans=True, random_state=42),
        )
        model_kmeans.fit(X_train, y_train)
        model_cmeans = dbc.DiscreteBayesianClassifier(
            discretizer=CMeansDiscretizer(init=model_kmeans.discretizer.centers_, fuzzifier=1.4),
        )
        model_cmeans.fit(X_train, y_train)
        
        def make_meshgrid(X, h=0.02, padding=0.5):
            x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
            y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
        
            xx, yy = np.meshgrid(
                np.arange(x_min, x_max, h),
                np.arange(y_min, y_max, h)
            )
            return xx, yy
        
        def predict_on_grid(model, xx, yy):
            X_grid = np.c_[xx.ravel(), yy.ravel()]
            y_grid = model.predict(X_grid)
            return y_grid.reshape(xx.shape)
        
        def predict_prob_on_grid(model, xx, yy, class_index=1):
            X_grid = np.c_[xx.ravel(), yy.ravel()]
            prob_grid = model.predict_prob(X_grid)
            prob_class = prob_grid[:, class_index]
            return prob_class.reshape(xx.shape)
        def compute_leaf_ids(tree_model, xx, yy):
            X_grid = np.c_[xx.ravel(), yy.ravel()]
            leaf_ids = tree_model.apply(X_grid)
            return leaf_ids.reshape(xx.shape)
        
        plt.rcParams.update({
            "pdf.fonttype": 42,        # Embed fonts as TrueType (safer for IEEE PDF checks)
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 9,
        })
        
        fig, ax = plt.subplots(
            1, 2,
            figsize=(7.16, 3.4),
            dpi=300,
            constrained_layout=True
        )
        
        xx, yy = make_meshgrid(X, h=0.005, padding=0.05)
        x_min, x_max = xx.min(), xx.max()
        y_min, y_max = yy.min(), yy.max()
        
        Z_Kmeans = predict_prob_on_grid(model_kmeans, xx, yy, class_index=1)
        Z_Cmeans = predict_prob_on_grid(model_cmeans, xx, yy, class_index=1)
        
        # cf0 = ax[0].contourf(xx, yy, Z_Kmeans, levels=30, alpha=0.85)
        # cf1 = ax[1].contourf(xx, yy, Z_Cmeans, levels=30, alpha=0.85)
        
        cf = ax[0].contourf(xx, yy, Z_Kmeans, levels=20, alpha=0.35, cmap='coolwarm')  # Decision regions
        ax[1].contourf(xx, yy, Z_Cmeans, levels=20, alpha=0.35, cmap='coolwarm')  # Decision regions
        
        # Voronoi overlay: make it subtle
        v0 = Voronoi(model_kmeans.discretizer.centers_)
        v1 = Voronoi(model_cmeans.discretizer.centers_)
        voronoi_plot_2d(v0, show_points=True, show_vertices=False, s=6, ax=ax[0], line_width=0.6, line_alpha=0.35)
        voronoi_plot_2d(v1, show_points=True, show_vertices=False, s=6, ax=ax[1], line_width=0.6, line_alpha=0.35)
        
        ax[0].contour(xx, yy, Z_Kmeans, levels=[0.5], colors="purple", linewidths=2)
        ax[1].contour(xx, yy, Z_Cmeans, levels=[0.5], colors="purple", linewidths=2)
        
        # One shared colorbar for probability (paper-grade quantitative mapping)
        cbar = fig.colorbar(cf, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_label("P(class = 1)", fontsize=8)
        cbar.ax.tick_params(labelsize=8)
        
        for a in ax:
            a.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors="k", alpha=0.7, s=15)  # Data points
        
            a.set_xlim(x_min, x_max)
            a.set_ylim(y_min, y_max)
            a.axis("off")
        
        
        ax[0].set_title("Hard discretization with K-means", fontsize=12)
        ax[1].set_title("Soft discretization with C-means", fontsize=12)
        
        fig.savefig(str(OUTPUT / 'soft_vs_hard_discritization.pdf'), bbox_inches="tight")
        from dbc.utils import compute_class_conditional_risk
        import dbc
        from dbc.discretizers import CMeansDiscretizer
        from sklearn.preprocessing import LabelEncoder
        
        # --- Train DBC (minmax=False) and DMC (minmax=True) ---
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        
        model_dbc = dbc.DiscreteBayesianClassifier(
            discretizer=CMeansDiscretizer(n_clusters=8, fuzzifier=1.3, use_kmeans=True, random_state=42),
            minmax=False,
        )
        model_dbc.fit(X_train, y_train)
        
        model_dmc = dbc.DiscreteBayesianClassifier(
            discretizer=CMeansDiscretizer(n_clusters=8, fuzzifier=1.3, use_kmeans=True, random_state=42),
            minmax=True,
            minmax_eps=0.001,
        
        )
        model_dmc.fit(X_train, y_train)
        
        # --- Compute class-conditional risks ---
        def get_risks(model, y_encoded):
            prior_pred = model.prior_best_ if model.minmax else model.prior_
            ccr = compute_class_conditional_risk(
                y=y_encoded,
                loss_function=model.loss_function_,
                p_hat=model.p_hat_,
                membership=model.membership_train_,
                prior_train=model.prior_,
                prior_pred=prior_pred,
            )
            global_risk = ccr @ prior_pred
            return ccr, global_risk
        
        ccr_dbc, gr_dbc = get_risks(model_dbc, y_train_encoded)
        ccr_dmc, gr_dmc = get_risks(model_dmc, y_train_encoded)
        
        # --- Plot 2x2 figure ---
        xx, yy = make_meshgrid(X, h=0.01, padding=-0.2)
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=300,
                                 gridspec_kw={'width_ratios': [2, 1]},
                                 constrained_layout=True)
        
        for row, (model, label, ccr, gr) in enumerate([
            (model_dbc, r'$h_\mathbf{\pi}$', ccr_dbc, gr_dbc),
            (model_dmc, r'$h_\mathbf{\pi^*}$', ccr_dmc, gr_dmc),
        ]):
            ax_boundary = axes[row, 0]
            ax_risk = axes[row, 1]
        
            # --- Left: Decision boundary ---
            Z = predict_prob_on_grid(model, xx, yy, class_index=1)
            cf = ax_boundary.contourf(xx, yy, Z, levels=20, alpha=0.35, cmap='coolwarm')
            ax_boundary.contour(xx, yy, Z, levels=[0.5], colors="purple", linewidths=2)
            ax_boundary.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm',
                                edgecolors="k", alpha=0.7, s=15)
            ax_boundary.set_xlim(xx.min(), xx.max())
            ax_boundary.set_ylim(yy.min(), yy.max())
            ax_boundary.axis("off")
            ax_boundary.set_title(f"{label} — Decision Boundary", fontsize=13)
        
            # --- Right: Risk bar chart ---
            labels_bar = ["$\hat{R}_{Blue}$", "$\hat{R}_{Red}$", "$\hat{R}$"]
            values = [ccr[0], ccr[1], gr]
            colors = ["#4393c3", "#d6604d", "#878787"]
            bars = ax_risk.bar(labels_bar, values, color=colors, edgecolor="k", width=0.55,)
            for bar, v in zip(bars, values):
                ax_risk.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                             f"{v:.3f}", ha="center", va="bottom", fontsize=11)
            ax_risk.set_ylabel("Risk", fontsize=13)
            ax_risk.set_ylim(0, max(ccr_dbc[0], ccr_dbc[1], gr_dbc, ccr_dmc[0], ccr_dmc[1], gr_dmc) * 1.2)
            ax_risk.set_title(f"{label} — Risks", fontsize=13)
            ax_risk.tick_params(axis="x", labelsize=13)
            # ax_risk.set_ylim()
        fig.savefig(str(OUTPUT / 'spdbc_vs_spdmc_comparison.pdf'), bbox_inches="tight")
        plt.show()
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, FancyBboxPatch
        
        # -----------------------------
        # Helper function
        # -----------------------------
        def gaussian_pdf(x, mu, sigma):
            """Return the 1D Gaussian density."""
            return 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        
        
        # -----------------------------
        # Data for the illustration
        # -----------------------------
        x = np.linspace(-1, 9, 1000)
        
        # Two class-conditional Gaussian densities
        mu1, sigma1 = 2.2, 1.0
        mu2, sigma2 = 5.2, 1.1
        y1 = gaussian_pdf(x, mu1, sigma1)
        y2 = gaussian_pdf(x, mu2, sigma2)
        
        # Segment boundaries
        bounds = [1.0, 3.2, 5.8]
        segments = [-1.0, bounds[0], bounds[1], bounds[2], 8.2]
        segment_labels = [r"$z_1$", r"$z_2$", r"$z_3$", r"$z_4$"]
        
        # Test point
        x_star = bounds[1]
        phi2 = 0.35
        phi3 = 0.65
        
        # -----------------------------
        # Figure setup
        # -----------------------------
        plt.rcParams.update({
            "font.size": 16,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral"
        })
        
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_xlim(-1.2, 8.8)
        ax.set_ylim(-0.9, 0.52)
        ax.axis("off")
        
        # -----------------------------
        # Background segments
        # -----------------------------
        segment_colors = ["#f3f3f3", "#eaf2fb", "#fdeeee", "#f3f3f3"]
        y_rect_bottom = -0.22
        rect_height = 0.22
        
        for i in range(4):
            left = segments[i]
            width = segments[i + 1] - segments[i]
            rect = Rectangle(
                (left, y_rect_bottom),
                width,
                rect_height,
                facecolor=segment_colors[i],
                edgecolor="none",
                alpha=0.9,
                zorder=0
            )
            ax.add_patch(rect)
        
        # -----------------------------
        # Main horizontal axis
        # -----------------------------
        ax.annotate(
            "",
            xy=(8.6, 0),
            xytext=(-1.05, 0),
            arrowprops=dict(arrowstyle="->", lw=2.2, color="black", mutation_scale=28),
            zorder=3
        )
        ax.text(8.9, -0.02, r"$x$", fontsize=34)
        
        # -----------------------------
        # Vertical dashed boundaries
        # -----------------------------
        for b in bounds:
            ax.plot([b, b], [0, y_rect_bottom + 0.01], linestyle=(0, (5, 4)), color="black", lw=2)
        
        # -----------------------------
        # Segment labels
        # -----------------------------
        for i in range(4):
            center = 0.5 * (segments[i] + segments[i + 1])
            ax.text(center, -0.07, segment_labels[i], ha="center", va="center", fontsize=28)
        
        # -----------------------------
        # Density curves
        # -----------------------------
        blue = "#0b4cc9"
        red = "#d91414"
        
        ax.plot(x, y1, color=blue, lw=3, label="Class 1", zorder=2)
        ax.plot(x, y2, color=red, lw=3, linestyle=(0, (3, 3)), label="Class 2", zorder=2)
        
        # Class labels at the top
        ax.plot([1.0, 1.5], [0.46, 0.46], color=blue, lw=3)
        ax.text(1.7, 0.46, "Class 1", color=blue, fontsize=24, va="center")
        
        ax.plot([4.3, 4.8], [0.46, 0.46], color=red, lw=3, linestyle=(0, (3, 3)))
        ax.text(5.0, 0.46, "Class 2", color=red, fontsize=24, va="center")
        
        # Gaussian labels
        ax.text(-0.1, 0.23, r"$\mathcal{N}\,(\mu_1,\sigma_1^2)$", color=blue, fontsize=28)
        ax.text(6.3, 0.23, r"$\mathcal{N}\,(\mu_2,\sigma_2^2)$", color=red, fontsize=28)
        
        # -----------------------------
        # Test point and label
        # -----------------------------
        ax.plot(x_star, -0.005, marker="*", color="black", markersize=26, zorder=5)
        ax.text(x_star - 0.10, -0.39, r"$x$", fontsize=38)
        
        # -----------------------------
        # Membership arrows
        # -----------------------------
        left_box_center = (2.2, -0.47)
        right_box_center = (4.35, -0.47)
        
        ax.annotate(
            "",
            xy=(left_box_center[0], left_box_center[1] + 0.06),
            xytext=(x_star, -0.01),
            arrowprops=dict(arrowstyle="-|>", lw=2, color=blue, mutation_scale=22)
        )
        
        ax.annotate(
            "",
            xy=(right_box_center[0], right_box_center[1] + 0.06),
            xytext=(x_star, -0.01),
            arrowprops=dict(arrowstyle="-|>", lw=2, color=red, mutation_scale=22)
        )
        
        # -----------------------------
        # Membership boxes
        # -----------------------------
        left_box = FancyBboxPatch(
            (1.25, -0.58), 1.55, 0.12,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            fc="white", ec=blue, lw=2
        )
        ax.add_patch(left_box)
        ax.text(2.025, -0.52, r"$\Phi_2(x)=0.5$", color=blue, fontsize=24, ha="center", va="center")
        
        right_box = FancyBboxPatch(
            (3.35, -0.58), 1.55, 0.12,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            fc="white", ec=red, lw=2
        )
        ax.add_patch(right_box)
        ax.text(4.125, -0.52, r"$\Phi_3(x)=0.5$", color=red, fontsize=24, ha="center", va="center")
        
        # -----------------------------
        # Formula box
        # -----------------------------
        formula_box = FancyBboxPatch(
            (5.25, -0.62), 3.25, 0.33,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            fc="white", ec="black", lw=2
        )
        ax.add_patch(formula_box)
        
        formula_text_1 = r"$\hat{P}(Y\mid x)\approx \sum_t \, \hat{P}(Y\mid Z=z_t)\Phi_t(x)$"
        formula_text_2 = r"$\approx 0.5\,\hat{P}(Y\mid z_2)+0.5\,\hat{P}(Y\mid z_3)$"
        
        ax.text(5.38, -0.42, formula_text_1, fontsize=22, ha="left", va="center")
        ax.text(5.38, -0.56, formula_text_2, fontsize=22, ha="left", va="center")
        
        # -----------------------------
        # Bottom banner
        # -----------------------------
        banner = FancyBboxPatch(
            (0.0, -0.86), 7.65, 0.11,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            fc="white", ec="#203a8f", lw=2
        )
        ax.add_patch(banner)
        
        ax.text(
            3.825, -0.805,
            "SPDBC combines profile-level posteriors using soft membership degrees.",
            color="#203a8f",
            fontsize=22,
            ha="center",
            va="center",
            style="italic"
        )
        
        # -----------------------------
        # Save and show
        # -----------------------------
        plt.tight_layout()
        plt.savefig(str(OUTPUT / 'spdbc_1d_illustration.png'), dpi=300, bbox_inches="tight")
        plt.savefig(str(OUTPUT / 'spdbc_1d_illustration.pdf'), bbox_inches="tight")
        plt.show()
        fig.savefig(str(OUTPUT / 'spdbc_1d_illustration.pdf'), bbox_inches="tight")

if __name__ == "__main__":
    main()
