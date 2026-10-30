<!-- Draft written with Claude Code (see Q1/assistance_log.md). Revise into your own words before submitting. Everything here repeats findings already reported in Sections 3 and 4 (NHSJS: no new information in the Conclusion). -->

5 Conclusion

5.1 Restatement of Key Findings

This study investigated how dropout affects the accuracy of a neural network and the mistakes it makes between similar clothing categories in Fashion-MNIST when the amount of training data is limited. When the models were trained long enough to overfit, dropout improved test accuracy at every training size from 1,000 to 10,000 images. Dropout 0.5 increased accuracy by between 0.53 and 0.95 percentage points, and this improvement was statistically significant at all four sizes. Dropout also made results more consistent across seeds and greatly reduced overfitting. However, dropout did not change the kinds of mistakes the network made. It reduced confusions between T-shirts, pullovers, coats, and shirts by about 1 percentage point, but at the same rate as it reduced all other mistakes, and shirt remained the hardest class in every condition.

5.2 Implications and Significance

These findings show that the benefit of dropout depends on whether a network is actually overfitting. My pilot study, which trained for only 10 epochs, found no benefit from dropout, while the final experiment, which trained until the models stopped improving, found a small but highly consistent benefit. This suggests that experiments comparing regularization methods can reach the wrong conclusion if training ends too early. The results also suggest that dropout is useful when training data is limited: with 1,000 images, it provided roughly 40% of the improvement gained by doubling the training set. Finally, the error analysis shows that dropout is not a solution for distinguishing visually similar categories. These confusions appear to come from the limits of the model and the images rather than from overfitting.

5.3 Connection to Objectives

My objectives were evaluated through four hypotheses. Hypothesis 1, that models without dropout overfit more with less training data, was mostly supported: the generalization gap was larger with 1,000 and 2,000 images than with 5,000 and 10,000. Hypothesis 2, that dropout reduces this gap, was not supported by the measure I planned in advance, because early stopping selected the models at very different points in training. The exploratory measures, which compared models at matched points in training, showed a large and consistent reduction. Hypothesis 3, that dropout reduces within-group confusion more with 10,000 images than with 2,000, was not supported: the reduction was small and about the same at both sizes. Hypothesis 4, that the gain from dropout depends on training size, was only partly supported, because the gain was smaller at 10,000 images but the difference was not statistically significant. Several of my original predictions were therefore not met. Even so, the reasons they were not met, including the role of early stopping and the source of the similar-class errors, were among the most useful results of the project.

5.4 Recommendations

Future studies should measure overfitting at matched points in training from the start, rather than only at the epoch selected by early stopping. Testing larger training sets, up to the full 60,000 images, would show whether the benefit of dropout keeps shrinking as data increases. Because the confusions between similar clothing categories did not respond to dropout, CNNs, which can detect local features such as collars and sleeves,<sup>10</sup> are a natural next step for studying those errors. Since the effect of dropout depends on the stage of training, schedules that apply dropout during only part of training<sup>8</sup> could also be compared. Testing learning rates below 0.003 for the dropout models would show whether the reported gains are an underestimate.

5.5 Limitations

This study used one dataset, one type of network (a two-hidden-layer MLP), and training sets of up to 10,000 images, so the results may not apply to other datasets, architectures, or larger amounts of data. All training sets were nested subsets drawn from a single fixed data split. Only two nonzero dropout rates were tested, and the dropout models used the lowest learning rate in the pilot grid. The accuracy improvements were small, under one percentage point. The exploratory overfitting measures were added after I saw the results, so they should be treated as supporting evidence rather than as tests of my original hypotheses.

5.6 Closing Thought

The difference between my pilot study and my final experiment was not dropout itself, but whether the experiment gave dropout a fair chance to work. A technique that looks useless under one setup can be clearly helpful under another, so how we test a method can matter as much as the method being tested.
