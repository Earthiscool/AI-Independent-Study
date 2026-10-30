<!-- Draft written with Claude Code (see Q1/assistance_log.md), built on my Week 4 research proposal. Revise into your own words before submitting. The hypotheses in 1.5 are copied word for word from configs/q1_dropout_protocol_v2.json and must stay that way. Citation numbers match Week6References.md. -->

1 Introduction

1.1 Background and Context

Neural networks learn to make predictions by adjusting large numbers of internal weights to fit a set of training examples. Because they are so flexible, they can also fit details that are specific to the training examples, such as noise or unusual images, instead of patterns that apply to new data. This problem is called overfitting. An overfit network performs very well on the images it was trained on but worse on images it has never seen, and the risk of overfitting is greatest when the training set is small.<sup>1</sup> Dropout is one of the most widely used methods for reducing overfitting.<sup>2,3</sup> This study examines how dropout affects not only how many mistakes a network makes, but which mistakes it makes, using a clothing-image dataset in which several categories look very similar.

1.2 Literature Review

1.2.1 Overfitting and Regularization

Techniques that reduce overfitting are called regularization methods.<sup>1</sup> Zhang et al. showed that standard image-classification networks have enough capacity to fit even randomly assigned labels, which means that a network's ability to perform well on new data cannot be explained by its size alone.<sup>4</sup> They also found that explicit regularization helped performance but was neither necessary nor sufficient for good generalization.<sup>4</sup> One simple and common form of regularization is early stopping, in which training is stopped when performance on a separate validation set stops improving. Prechelt studied how to choose a stopping rule systematically and showed that early stopping involves a trade-off between training time and generalization.<sup>5</sup>

1.2.2 Dropout

Dropout was introduced by Hinton et al., who showed that randomly omitting about half of the hidden units on each training example greatly reduced overfitting.<sup>2</sup> Srivastava et al. later described the method in detail and tested it on vision, speech recognition, document classification, and computational biology tasks.<sup>3</sup> They gave two explanations for why dropout works. First, because any unit may be removed, units cannot "co-adapt," or depend on specific other units to correct their mistakes, so each unit must learn features that are useful on their own. Second, training with dropout samples from an exponential number of smaller "thinned" networks that share weights, and using the full network at test time approximately averages their predictions.<sup>3</sup>

Later research has offered other ways to understand dropout. Wager et al. showed that, for generalized linear models, dropout acts as a form of adaptive regularization that penalizes large weights in a way that depends on the data.<sup>6</sup> Gal and Ghahramani showed that dropout can be interpreted as an approximation to Bayesian inference, which allows a dropout network to estimate how uncertain its predictions are.<sup>7</sup> More recently, Liu et al. found that dropout can also help models that are underfitting when it is used only early in training, showing that the effect of dropout depends on the stage of training at which it is applied.<sup>8</sup>

1.2.3 Dropout and the Amount of Training Data

Srivastava et al. directly tested how the size of the training set changes the benefit of dropout.<sup>3</sup> They trained a large network (784-1024-1024-2048-10) on random subsets of the MNIST handwritten-digit dataset containing between 100 and 50,000 images. For extremely small training sets (100 and 500 images), dropout gave no improvement, because the network could still memorize the training data despite the noise from dropout. As the training set grew, the gain from dropout increased up to a point and then declined. The authors suggested that, for a given network and dropout rate, there is a "sweet spot": an amount of data large enough that it cannot simply be memorized, but not so large that overfitting stops being a problem.<sup>3</sup> This experiment measured only the overall error rate, so it did not show whether dropout changed the kinds of mistakes the network made at different training sizes.

1.2.4 Fashion-MNIST and Similar Clothing Categories

Fashion-MNIST was introduced by Xiao et al. as a more challenging replacement for MNIST.<sup>9</sup> Like MNIST,<sup>10</sup> it contains 70,000 grayscale images of 28 × 28 pixels divided into 60,000 training and 10,000 test images, but its ten categories are clothing items instead of digits. Several of these categories are visually similar. Nocentini et al. trained multiple convolutional neural networks (CNNs) on Fashion-MNIST, reaching about 94% accuracy, and found that shirts were the hardest class to recognize because they were mostly confused with T-shirts/tops.<sup>11</sup> Hobbs and Shah similarly found that the errors of their CNN clustered among upper-body garments (shirts, T-shirts, coats, and pullovers) and attributed this mainly to the low resolution of the images.<sup>12</sup> These studies suggest that confusions among these four categories are a persistent problem even for strong models.

1.2.5 Evaluating Classifiers Beyond Overall Accuracy

Overall accuracy summarizes performance with a single number, but it does not show how errors are distributed across classes. Sokolova and Lapalme reviewed performance measures for multi-class classification, including per-class precision and recall and macro-averaged measures that give each class equal weight.<sup>13</sup> Balestriero et al. showed why this matters for regularization: regularization methods such as data augmentation and weight decay can improve average accuracy while greatly reducing accuracy on particular classes, meaning that the effects of regularization are class-dependent.<sup>14</sup> Comparisons between training methods are also affected by random variation. Bouthillier et al. showed that random factors such as weight initialization and data order can change benchmark results enough to hide or create apparent improvements, and recommended accounting for these sources of variation by repeating experiments.<sup>15</sup> Dietterich similarly showed that the choice of statistical test strongly affects how reliably two learning methods can be compared.<sup>16</sup>

1.2.6 Gap in the Literature

Together, these studies show that dropout reduces overfitting,<sup>2,3</sup> that its overall benefit depends on the amount of training data,<sup>3</sup> that the effects of regularization can differ between classes,<sup>14</sup> and that some Fashion-MNIST categories are consistently confused with each other.<sup>11,12</sup> However, none of the studies reviewed here examined whether dropout changes the specific confusions between visually similar categories, or whether that effect depends on the size of the training set. Answering this question also requires comparisons that account for random variation between training runs.<sup>15</sup>

1.3 Problem Statement and Rationale

Dropout is usually evaluated by its effect on overall accuracy, but two networks with the same accuracy can make very different errors. For Fashion-MNIST, an important question is whether dropout helps a network separate the similar clothing categories that cause many of its errors, or whether it only reduces mistakes in general. Because dropout is meant to reduce overfitting, and overfitting depends on the size of the training set, the answer may also change as more training data becomes available. In this study, I investigate both questions together.

1.4 Significance and Purpose

In many real applications, collecting and labeling training data is expensive, so models must often be trained on limited data. Knowing how much dropout helps at different training sizes, and which kinds of mistakes it does or does not fix, can help decide when dropout is worth using and when a different approach is needed. If dropout reduces confusions between similar categories, it could be a simple way to improve a model on fine-grained distinctions; if it does not, those errors need other solutions. This study also aims to measure the effect of dropout carefully, using repeated runs, a held-out test set, and statistical tests, so that small effects can be separated from random variation between training runs.

1.5 Objectives

The research question for this study is: How does dropout affect the overall accuracy of a neural network and the mistakes it makes when identifying similar clothing categories in Fashion-MNIST, and how do these effects change with the amount of training data?

Before running the final experiment, I recorded four hypotheses:

- H1: Without dropout, the gap between training and validation accuracy grows as the training set gets smaller.
- H2: Dropout reduces that gap at every training size.
- H3: Dropout lowers the within-group confusion rate for T-shirt/top, pullover, coat and shirt, and the reduction is larger at 10,000 images than at 2,000 images.
- H4: The test-accuracy gain from dropout depends on training-set size.

1.6 Scope and Limitations

This study focuses on one dataset (Fashion-MNIST) and one type of neural network, a multilayer perceptron (MLP) with two hidden layers. It compares three dropout rates (0.0, 0.2, and 0.5) and four training set sizes between 1,000 and 10,000 images. The analysis of similar categories is limited to four classes: T-shirt/top, pullover, coat, and shirt. The study does not examine CNNs, other regularization methods such as weight decay or data augmentation, or training sets larger than 10,000 images, so the results may not apply directly to other datasets, architectures, or much larger amounts of training data.

1.7 Theoretical Framework

This study is based on the idea of the generalization gap, the difference between how well a model performs on its training data and on new data.<sup>1</sup> A model with too much flexibility for its amount of training data can memorize the training set and develop a large gap, and regularization methods aim to reduce this gap. Both of Srivastava et al.'s explanations of dropout, preventing co-adaptation and approximately averaging many thinned networks, predict that dropout should help most when a network is overfitting, which in turn depends on the size of the training set.<sup>3</sup> This framework guided the design of the study: the effect of dropout should be measured under conditions where the network is able to overfit, and its benefit should change with training size. It also raises the central question of this study, which is whether reducing overfitting is enough to fix confusions between similar categories, or whether those confusions have a different cause.

1.8 Methodology Overview

I conducted a controlled computational experiment in which I trained an MLP on Fashion-MNIST while varying the dropout rate and the size of the training set. Each of the 12 conditions was repeated with 10 random seeds, and the models were trained with early stopping. After all training was complete, the models were evaluated once on the official Fashion-MNIST test set. I compared overall accuracy, overfitting, and errors between the four similar clothing categories using paired statistical tests. This experiment followed a smaller pilot study, whose results led to the final design. The full methodology is described in Section 2.
