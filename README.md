# Dotplot Technical Exercise — Approach Note

## Dataset
Breast Cancer Wisconsin (Original), UCI Machine Learning Repository, 699 cases,
9 integer-valued cytological attributes plus an ID and Attribute 11 (Class:
2 = benign, 4 = malignant). Retrieved via the official `ucimlrepo` PyPI
package (`fetch_ucirepo(id=15)`), using `.data.original` to get a single
DataFrame with ID, all nine features, and Class already aligned — this
requires live internet access to archive.ics.uci.edu at runtime, in
exchange for not having to hand-parse the raw `.data` file.

## Pipeline (`BreastCancerPrediction.py`)
Everything runs in-memory with pandas; no database or persisted intermediate
dataset is created — only the final CSV and chart are written to
`output_Dotplot_breast_cancer/`.

1. **Ingest**: fetched via `ucimlrepo` as described above. Missing values
   (`Bare_nuclei` only) arrive already as proper `NaN`, so no manual `"?"`
   parsing is needed.
2. **Prepare**:
   - Attribute 11 (Class) is split off immediately and never touches the
     feature matrix used for scoring — it is only reattached at export time
     for the correlation check.
   - Removed every row where `Clump_Thickness == 1` (145 of 699 rows), per
     the brief.
   - **Missing-value policy**: only `Bare_Nuclei` contains missing values in
     this dataset (16 raw rows). Rather than imputing, I **dropped** rows
     with a missing feature value (10 rows remaining after the Clump
     Thickness filter). Rationale: the missing fraction is small (<3% of
     the original data), the dataset is otherwise complete and small enough
     that dropping doesn't meaningfully hurt statistical power, and this
     avoids injecting an imputed value (e.g. a column median) into the very
     attribute the malignant-likelihood score depends on.
   - 544 cases remain after both filters.
3. **Score**: fit K-means (k=2) on the nine standardised cytological
   attributes (Clump Thickness, Uniformity of Cell Size/Shape, Marginal
   Adhesion, Single Epithelial Cell Size, Bare Nuclei, Bland Chromatin,
   Normal Nucleoli, Mitoses), then converted the clustering into a
   continuous score using each case's **relative distance to the two
   cluster centroids**.
   - **Why these attributes**: all nine are cytological measurements on a
     1–10 scale where, per the dataset documentation, a higher value
     indicates a more abnormal-looking cell. Using all nine (rather than an
     arbitrary pair) lets the clustering summarise the overall abnormality
     pattern rather than depending on which two attributes were picked.
   - **Why K-means**: a standard, simple, fully unsupervised way to find
     natural groupings in the feature space, with no reference to
     Attribute 11 anywhere in its construction. With k=2 the two clusters
     (sizes 323 / 221) roughly correspond to "typical-looking" and
     "atypical-looking" cytology, discovered purely from how the nine
     attributes co-vary.
   - **Cluster labels → continuous score**: a raw cluster ID is only ever
     two values, which doesn't satisfy "a higher score should represent a
     higher likelihood." Instead, for each case we compute its Euclidean
     distance (in standardised feature space) to both centroids, and score
     it as:
     ```
     score = distance_to_low_centroid / (distance_to_low_centroid + distance_to_high_centroid) * 100
     ```
     A case sitting exactly on the "low abnormality" centroid scores 0; one
     sitting exactly on the "high abnormality" centroid scores 100;
     everything else falls smoothly in between based on relative
     proximity. In practice no case coincides exactly with either
     centroid, so the score naturally spans roughly **6.6 to 81.5** rather
     than the full 0–100 — this is expected behaviour of a distance-ratio
     score, not a rescaling step, and no min-max stretching is applied
     afterwards.
   - **Orienting the two centroids**: K-means doesn't know which cluster is
     "malignant-looking" — cluster indices are arbitrary. The "high
     abnormality" centroid is identified using only feature magnitude
     (never the label): whichever centroid has the higher mean
     standardised value across all nine attributes is treated as "high".
4. **Compare with Attribute 11** (only at this final step, for reporting):
   - **Pearson correlation (point-biserial)**: r = **0.925**. This is the
     standard, primary measure for a continuous score against a binary
     outcome, and is straightforward to interpret.
   - **Spearman rank correlation**: ρ = **0.842**, reported as a robustness
     check since it only assumes a monotonic (not linear) relationship.
   - **ROC-AUC**: **0.990**, reported as a third, complementary measure.
     Correlation answers "how strongly do score and outcome move
     together?"; AUC instead asks a more clinically meaningful question:
     "if I pick one random malignant case and one random benign case, how
     often does the malignant one get the higher score?" It treats the
     score purely as a ranking function, needs no linearity assumption,
     and is visualised directly via the ROC curve in `score_summary.png`.
     As with the other two measures, Attribute 11 is only ever used here
     to *grade* a score that was built entirely without it.
   - All three measures are strongly positive and highly significant
     (p ≈ 0), suggesting the unsupervised score tracks the pathologist's
     diagnosis well, even though the diagnosis was never used to build it.
5. **Export**: `scored_data.csv` — one row per retained case, with `Sample_code_number`, all
   nine input attributes, `Malignant_Score`, and `Class`, so every score is
   traceable back to its source record.
6. **Visualisation**: `score_summary.png` — three panels: a histogram and
   a boxplot of the score split by true Class, plus an ROC curve showing
   the score's discrimination quality as a ranking function.

## Reproducing
Please add your output address in "OUT_DIR"
