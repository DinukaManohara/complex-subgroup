# `Adressing R1.D1-D3, D6; R3.D4 - Table and figures for the ICDE 2027 rebuttal

## Background

To address the concern about picking the best over several runs by R1, we re-ran SCOUT on all datasets with deterministic initialisation and averaged exceptionality over all (β, λ) combinations for each pocket size, then selected the best average per dataset. Those results are presented on the table `00_rebuttal_table.pdf`. In Figure 5, each plotted point is not a single parameter combination: it is the average over all remaining parameters with the x-axis value held fixed (e.g., for a fixed β, we average over all p and λ combinations). This aggregation, rather than the initialisation, is a contributing reason for the apparent clutter - the plots for the default-initialised runs behave the same way. To avoid this confusion, we additionally present heatmaps for all datasets (link), separated by pocket size (p), with β and λ on the x- and y-axes and color encoding the JSD or the component count.

## Contents

**β × λ heatmaps (parameter sensitivity)**
`beta_lambda_<dataset>_score_jsd.pdf` - one file per dataset. Panels run left
to right over pocket size `p`; within a panel the x-axis is β, the y-axis is λ,
and colour is the mean JSD over α. The white star marks the best cell in each
panel. The colour scale is shared across a dataset's panels, so the panels can
be compared to each other but not across datasets. The component-count counterpart,
`beta_lambda_<dataset>_num_clusters_jsd.pdf`.

**Line charts - default-initialisation runs**
`parameter_all_trends.pdf` - The six-panel trend figure: JSD and average
component count against `log2(p)`, β, and λ, one line per dataset, averaged across all parameter combinations (while one is fixed) for the new runs under the default (deterministic) initialisation.

**Result table - divergence values**
`00_rebuttal_table.pdf` - divergence values for the same set of new experiments.

**Parameter trends (existing)**
`parameter_all_trends.pdf` - the six-panel trend figure: JSD and average
component count against `log2(p)`, β, and λ, one line per dataset.


