<!-- Draft written with Claude Code (see Q1/assistance_log.md). Every number comes from Q1/results/raw/v2/. Revise into your own words before submitting. -->

3 Results

3.1 Pilot Study

In the Week 5 pilot study, dropout did not improve validation accuracy. With 2,000 training images, the mean validation accuracy was 77.83% without dropout and 76.73% with dropout 0.3; with 10,000 images, it was 83.34% and 82.87%. None of these differences were statistically significant with three seeds. As described in Section 2.1, the pilot models were still improving when training ended, so these results motivated the redesigned experiment reported below.

3.2 Training Behavior

All 120 final runs stopped early, before reaching the 150-epoch limit, and none diverged. Models without dropout reached their lowest validation loss early in training (median best epoch 13 to 17, depending on training size), while models with dropout took longer (median 33.5 to 49.5 epochs with dropout 0.2 and 43.5 to 65 epochs with dropout 0.5).

Figure 1 shows the average loss curves. Without dropout, training loss kept falling while validation loss reached its minimum and then rose, which is the typical pattern of overfitting. With dropout, validation loss decreased more slowly but stayed almost flat after reaching its minimum.

Figure 1. Mean training loss (dotted) and validation loss (solid) across 10 seeds for each training size and dropout rate. Shaded areas show the range across seeds. Runs stop at different epochs because of early stopping, so later epochs average only the runs that were still training. (`q1v2_loss_curves.png`)

3.3 Test Accuracy

Dropout improved test accuracy at every training size (Table 2, Figure 2). Accuracy increased with training size for every dropout rate, from about 79–80% with 1,000 images to about 86% with 10,000 images. At every size, dropout 0.5 had the highest mean accuracy, followed by dropout 0.2, then no dropout. Results with dropout were also more consistent across seeds: the standard deviation of test accuracy was smaller with dropout than without it in all eight comparisons.

| Training images | No dropout | Dropout 0.2 | Dropout 0.5 |
|---|---|---|---|
| 1,000 | 79.05 ± 0.55 (0.789) | 79.78 ± 0.25 (0.796) | 79.93 ± 0.28 (0.798) |
| 2,000 | 81.36 ± 0.60 (0.814) | 81.97 ± 0.26 (0.819) | 82.31 ± 0.36 (0.823) |
| 5,000 | 83.96 ± 0.50 (0.838) | 84.55 ± 0.20 (0.844) | 84.80 ± 0.31 (0.847) |
| 10,000 | 85.62 ± 0.46 (0.855) | 86.03 ± 0.38 (0.860) | 86.15 ± 0.24 (0.861) |

Table 2. Test accuracy (%) as mean ± standard deviation over 10 seeds, with mean macro-F1 in parentheses.

Table 3 shows the paired comparisons. Dropout 0.5 improved test accuracy by between 0.53 and 0.95 percentage points, and this improvement was statistically significant at all four training sizes after Holm correction. Dropout 0.2 improved accuracy by between 0.41 and 0.74 points and was significant at 1,000, 2,000, and 5,000 images, but not at 10,000 images, where the 95% confidence interval included zero. In total, 7 of the 8 accuracy comparisons were significant. The improvement was also consistent at the level of individual seeds: dropout 0.5 outperformed its no-dropout partner in all 10 seeds at 1,000 and 2,000 images and in 9 of 10 seeds at 5,000 and 10,000 images.

| Training images | Dropout 0.2: gain [95% CI] | Holm p | Seeds improved | Dropout 0.5: gain [95% CI] | Holm p | Seeds improved |
|---|---|---|---|---|---|---|
| 1,000 | +0.74 [0.19, 1.28] | 0.029 | 9/10 | +0.88 [0.46, 1.30] | 0.008 | 10/10 |
| 2,000 | +0.61 [0.24, 0.98] | 0.019 | 9/10 | +0.95 [0.54, 1.36] | 0.004 | 10/10 |
| 5,000 | +0.59 [0.25, 0.94] | 0.018 | 9/10 | +0.84 [0.43, 1.25] | 0.008 | 9/10 |
| 10,000 | +0.41 [−0.07, 0.89] | 0.088 | 7/10 | +0.53 [0.16, 0.90] | 0.029 | 9/10 |

Table 3. Change in test accuracy (percentage points) compared with no dropout, paired by seed.

Macro-F1 showed the same direction of effect. Dropout increased macro-F1 by between 0.005 and 0.009 in all eight comparisons, and 5 of the 8 increases were significant after Holm correction.

Figure 2. Left: test accuracy versus training size for each dropout rate (mean ± 95% CI over 10 seeds). Right: accuracy gain from dropout compared with no dropout, paired by seed (mean ± 95% CI); the horizontal line marks no change. (`q1v2_accuracy_vs_size.png`)

3.4 Overfitting (Hypotheses 1 and 2)

Hypothesis 1 predicted that, without dropout, the generalization gap would grow as the training set got smaller. Using the pre-registered measure (training accuracy minus validation accuracy at the selected epoch), the gap for models without dropout was 9.90 points with 1,000 images, 10.03 with 2,000, 7.30 with 5,000, and 5.71 with 10,000. The gap was therefore larger for the two smallest training sizes than for the two largest, although it was nearly the same at 1,000 and 2,000 images.

Hypothesis 2 predicted that dropout would reduce this gap. Using the pre-registered measure, it did not: the differences between dropout and no-dropout models ranged from −0.09 to +1.12 points, and none were significant after Holm correction. This measure compares models at the epoch each one was selected, which was around epoch 15 for models without dropout but around epoch 35 to 65 for models with dropout. Because the no-dropout models were selected before most of their overfitting occurred (Figure 1), this measure does not capture how much each model overfit as training continued.

The two exploratory measures, added after seeing the results, compare the models at matched points in training (Table 4, Figure 3). At epoch 20, the generalization gap without dropout ranged from 12.13 points (1,000 images) to 7.36 points (10,000 images), decreasing steadily as training size increased. With dropout, the gap stayed between 3.75 and 5.42 points at every size. Dropout reduced the gap in all 80 seed pairs, by between 3.0 and 8.3 points on average, with the largest reduction at 1,000 images. Similarly, in the 15 epochs after reaching their best epoch, the validation loss of models without dropout rose by about 0.10 to 0.11, while models with dropout rose by only 0.016 to 0.035. All 16 exploratory comparisons were significant after Holm correction (largest adjusted p = 0.007).

| Training images | Gap at epoch 20: no dropout | Dropout 0.2 | Dropout 0.5 | Val-loss rise: no dropout | Dropout 0.2 | Dropout 0.5 |
|---|---|---|---|---|---|---|
| 1,000 | 12.13 | 4.16 | 3.84 | 0.101 | 0.024 | 0.021 |
| 2,000 | 10.25 | 5.42 | 4.38 | 0.104 | 0.033 | 0.019 |
| 5,000 | 8.39 | 4.66 | 3.99 | 0.111 | 0.035 | 0.020 |
| 10,000 | 7.36 | 4.32 | 3.75 | 0.108 | 0.032 | 0.016 |

Table 4. Exploratory (not pre-registered) overfitting measures, mean over 10 seeds. The gap is training accuracy minus validation accuracy (percentage points) at epoch 20; the validation-loss rise is the increase from each model's best epoch to the end of its training.

Figure 3. Exploratory overfitting measures versus training size (mean ± 95% CI over 10 seeds). Left: generalization gap at epoch 20. Right: rise in validation loss during the 15 epochs after the best epoch. (`q1v2_exploratory_overfitting.png`)

3.5 Errors Between Similar Clothing Categories (Hypothesis 3)

Across all conditions, shirt was the hardest class: its test recall was between 48% and 62%, the lowest of all ten classes. The most common mistakes within the group of four similar classes were shirts predicted as T-shirts/tops (14–18% of shirts) and pullovers and coats predicted as each other (10–15%). These patterns looked similar with and without dropout (Figure 4).

Dropout lowered the within-group confusion rate at every training size (Table 5), by between 0.82 and 1.26 percentage points. All eight reductions had uncorrected p-values below 0.05, but after Holm correction only one remained significant (5,000 images with dropout 0.5: −1.02 points, adjusted p = 0.039).

| Training images | Within-group confusion (%): 0.0 / 0.2 / 0.5 | Within-group error share (%): 0.0 / 0.2 / 0.5 | Shirt recall (%): 0.0 / 0.2 / 0.5 |
|---|---|---|---|
| 1,000 | 28.5 / 27.3 / 27.3 | 82.3 / 81.3 / 81.8 | 48.3 / 47.9 / 49.2 |
| 2,000 | 26.5 / 25.4 / 25.3 | 84.3 / 82.3 / 83.8 | 57.1 / 54.4 / 54.8 |
| 5,000 | 23.0 / 22.2 / 22.0 | 81.2 / 82.3 / 82.0 | 54.1 / 56.1 / 57.4 |
| 10,000 | 21.8 / 20.9 / 20.9 | 84.3 / 83.8 / 82.8 | 57.9 / 61.9 / 61.5 |

Table 5. Similar-class error measures on the test set by dropout rate (0.0 / 0.2 / 0.5), mean over 10 seeds.

The within-group error share did not change with dropout. About 81–84% of the mistakes on these four classes were confusions with another class in the group, regardless of dropout rate, and none of the differences were significant (all adjusted p = 1.0). This means that dropout reduced mistakes between similar classes at about the same rate as it reduced other mistakes, rather than specifically changing the type of mistakes the model made. Shirt recall also showed no significant change with dropout (differences from −2.7 to +4.0 points, all adjusted p ≥ 0.59), so the pilot study's observation that dropout made the model avoid predicting "shirt" was not repeated.

Hypothesis 3 also predicted that the reduction in within-group confusion would be larger with 10,000 images than with 2,000. This was not supported. The reduction was slightly smaller at 10,000 images (−0.85 vs −1.08 points for dropout 0.2; −0.92 vs −1.26 for dropout 0.5), and the differences between sizes were not significant (Welch's t-test, p = 0.65 and p = 0.56).

Figure 4. Test-set confusion among T-shirt/top, pullover, coat, and shirt with 1,000 and 10,000 training images, summed over 10 seeds. Each cell shows the percentage of images of the actual class (row) predicted as each class (column). (`q1v2_similar_classes.png`)

3.6 Effect of Training Size (Hypothesis 4)

Hypothesis 4 predicted that the accuracy gain from dropout would depend on training-set size. The gain was smaller at 10,000 images than at 2,000 images for both dropout rates (+0.41 vs +0.61 points for dropout 0.2; +0.53 vs +0.95 points for dropout 0.5), and Figure 2 shows the gain staying roughly level from 1,000 to 5,000 images before declining at 10,000. However, the difference between 2,000 and 10,000 images was not statistically significant for either dropout rate (Welch's t-test, p = 0.47 for dropout 0.2 and p = 0.10 for dropout 0.5). The results are therefore consistent with a smaller benefit from dropout as training data increases, but this experiment did not provide statistically significant evidence for it.
