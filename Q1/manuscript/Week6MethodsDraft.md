<!-- Draft written with Claude Code (see Q1/assistance_log.md). Revise into your own words before submitting. -->

2 Methodology

2.1 Research Design

This study was a controlled computational experiment. I trained a neural network to classify Fashion-MNIST images and changed two factors: the dropout rate (0.0, 0.2, or 0.5) and the number of training images (1,000, 2,000, 5,000, or 10,000). Crossing these factors produced 12 experimental conditions. I trained each condition with 10 different random seeds, for a total of 120 final training runs, because random factors such as weight initialization and data order can change results enough to hide or exaggerate small differences.<sup>15</sup>

The design is paired. For a given seed, the models with and without dropout started from the same initial weights and saw the training images in the same order. Because of this, I could compare each dropout model directly with its own no-dropout partner instead of only comparing group averages.

The final experiment was based on an earlier pilot study that I ran in Week 5. In the pilot, I trained a smaller network (784-128-64-10) with dropout rates of 0.0 and 0.3 on 2,000 and 10,000 images, using three seeds, a learning rate of 0.1, and a fixed budget of 10 epochs. Dropout did not improve validation accuracy at either training size. However, the pilot's training curves showed that the models had not finished learning: 11 of the 12 runs reached their lowest validation loss at epoch 9 or 10, and the gap between training and validation accuracy for models without dropout was only about 4 percentage points. Since dropout is designed to reduce overfitting, and the pilot models were not yet overfitting, the pilot could not fairly test whether dropout helps. I therefore redesigned the experiment to train longer with early stopping, use a wider network, test more dropout rates and training sizes, use more seeds, and report results on the official test set.

Before the final runs, I saved the full experimental protocol, including my hypotheses, all settings, and the planned statistical analysis, to a configuration file (`q1_dropout_protocol_v2.json`). The code refuses to continue if any setting in this file is changed afterward, so the analysis plan could not be adjusted after seeing the results.

2.2 Participants or Sample

This study did not involve human participants. The sample was the Fashion-MNIST dataset introduced by Xiao et al.<sup>9</sup> Fashion-MNIST contains 70,000 grayscale images of clothing items, each 28 × 28 pixels, divided into 60,000 training images and 10,000 official test images. Each image belongs to one of ten categories: T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, and ankle boot.

I used three separate groups of images:

- Training subsets. In Week 1, I shuffled the 60,000 original training images using a fixed seed of 42 and saved the resulting indices. From this shuffled order, I took a fixed set of 10,000 training images. The smaller training sets were the first 1,000, 2,000, and 5,000 images of this same set, so each smaller set was fully contained in the larger ones. This nesting means the only difference between training sizes is the number of additional examples. Each class was represented in every subset (for example, between 90 and 118 images per class in the 1,000-image subset and between 938 and 1,052 per class in the 10,000-image subset).
- Validation set. I reserved 5,000 different images from the original training data (between 465 and 546 per class). These were used only to choose the learning rate and to decide when to stop training. I confirmed that none of these images overlapped with the training images.
- Test set. The official 10,000-image Fashion-MNIST test set, with exactly 1,000 images per class, was used only once, after all 120 final runs were complete, to measure the reported results.

Pixel values were scaled from the range 0 to 255 to the range 0 to 1, which matches PyTorch's `transforms.ToTensor()`. No other preprocessing or data augmentation was used.

2.3 Data Collection

In this experiment, each training run is one observation. For every run, my code recorded the training and validation measurements after each epoch, saved the model weights from the selected epoch, and saved the run's final measurements to a CSV file. After all runs were complete, the code evaluated each saved model on the test set and saved its predictions, confusion matrix, and per-class precision, recall, and F1 scores.

All runs were performed on an Apple M4 computer using Python 3.14.0, PyTorch 2.14.0,<sup>17</sup> and torchvision 0.29.0. I ran the experiment on the CPU instead of Apple's MPS graphics processor because, for a network of this size, a timing test showed the CPU was about five times faster per epoch, and CPU computations give exactly repeatable results. All code, settings, data split indices, and raw results are saved in my project repository so that every reported number can be traced back to the run that produced it.

2.4 Variables and Measurements

| Type | Variable | Values or definition |
|---|---|---|
| Independent | Dropout rate | 0.0 (no dropout), 0.2, 0.5 |
| Independent | Training set size | 1,000, 2,000, 5,000, 10,000 images |
| Controlled | Network architecture | 784-512-256-10 MLP with ReLU |
| Controlled | Optimizer and batch size | SGD with momentum 0.9, batch size 32 |
| Controlled | Data split | Same training, validation, and test images for every run |
| Controlled | Initialization and batch order | Identical for all dropout rates within a seed |
| Dependent (primary) | Test accuracy | Percentage of the 10,000 test images classified correctly |
| Dependent | Macro-F1 | Average F1 score across all ten classes |
| Dependent | Within-group confusion rate | See below |
| Dependent | Within-group error share | See below |
| Dependent | Shirt recall | Percentage of real shirts predicted as "Shirt" |
| Dependent | Generalization gap | Training accuracy minus validation accuracy at the selected epoch |

Test accuracy was my primary measure because it summarizes overall performance on images the model never saw during training or model selection. Precision measures how often a prediction for a class is correct, and recall measures how many real examples of a class the model identifies. The F1 score combines precision and recall, and macro-F1 averages F1 across all ten classes so that each class counts equally.<sup>13</sup>

To study mistakes between visually similar clothing, I focused on four classes: T-shirt/top, pullover, coat, and shirt. I measured these mistakes in two ways:

- The within-group confusion rate is the number of images from these four classes that were predicted as a different class within the same group, divided by the total number of images from the four classes, multiplied by 100. This is the same measure I used in the pilot study.
- The within-group error share is the number of those within-group mistakes divided by all mistakes made on images from the four classes, multiplied by 100. This separates two possibilities: dropout could reduce mistakes overall, or it could specifically change what kind of mistakes the model makes. If dropout only reduces mistakes overall, the confusion rate should fall but the error share should stay about the same.

I also recorded shirt recall because shirt was the hardest class in the pilot study, and in the pilot, dropout appeared to make the model predict "Shirt" less often.

The generalization gap measures overfitting. A model that has memorized its training images will have much higher training accuracy than validation accuracy. To make this comparison fair, I measured training accuracy with dropout turned off, in the same way as validation accuracy. In the pilot study, training accuracy had been measured with dropout active, which made the dropout models look worse on their own training data.

2.5 Procedure

Step 1: Network. I used a multilayer perceptron (MLP). Each 28 × 28 image was flattened into 784 values and passed through two hidden layers of 512 and 256 neurons, each followed by a ReLU activation function, and then an output layer of 10 neurons, one per class. The network has 535,818 trainable parameters. I made the network wider than the 128-64 network used in the pilot because a larger network has more capacity to memorize training images, which is the situation dropout is designed for. Weights were initialized with PyTorch's default method.

Step 2: Dropout. In the dropout models, a dropout layer followed each hidden layer's ReLU activation, the same placement as in the pilot study. During training, each hidden activation was set to zero with probability p (0.2 or 0.5), and the remaining activations were scaled up by 1/(1 − p) so the average signal size stayed the same. Dropout was turned off whenever the model was evaluated. I tested 0.5 because it is the hidden-layer rate used by Srivastava et al.,<sup>3</sup> and 0.2 as a milder setting. The no-dropout model was identical except that p = 0.

Step 3: Training. Models were trained with stochastic gradient descent (SGD) with momentum 0.9<sup>18</sup> and a batch size of 32, using cross-entropy loss. The training images were shuffled at the start of every epoch using a random number generator tied to the run's seed. After each epoch, I computed the training loss and accuracy (with dropout off) and the validation loss and accuracy.

Step 4: Early stopping. Instead of training for a fixed number of epochs, each model trained for up to 150 epochs and stopped when its validation loss had not improved for 15 consecutive epochs.<sup>5</sup> The model weights from the epoch with the lowest validation loss were kept as the final model for that run. This allowed each model to train until it stopped improving, which solved the main problem found in the pilot study.

Step 5: Learning-rate pilot. Dropout can change which learning rate works best, so instead of using one learning rate for every model, I selected a learning rate separately for each dropout rate. I tested learning rates of 0.003, 0.01, 0.03, and 0.1 for each dropout rate at training sizes of 2,000 and 10,000, using two seeds (0 and 1) that were not used in the final experiment. This made 48 pilot runs. Each pilot run used the same early-stopping procedure, and I chose the learning rate with the lowest average validation loss across its four pilot runs. Only the validation set was used.

| Dropout rate | lr 0.003 | lr 0.01 | lr 0.03 | lr 0.1 | Selected |
|---|---|---|---|---|---|
| 0.0 | 0.4714 | 0.4664 | 0.4833 | 0.6673 | 0.01 |
| 0.2 | 0.4500 | 0.4558 | 0.4767 | 0.7296 | 0.003 |
| 0.5 | 0.4420 | 0.4515 | 0.5237 | 1.2463 | 0.003 |

Table 1. Mean lowest validation loss in the learning-rate pilot (lower is better).

Step 6: Freezing the protocol. After the learning-rate pilot, I saved all settings, the selected learning rates, my hypotheses, and the analysis plan to the protocol file described in Section 2.1.

Step 7: Final runs. I trained all 120 final models (3 dropout rates × 4 training sizes × 10 seeds: 42, 123, 456, 789, 1011, 2024, 3141, 4242, 5555, and 6789).

Step 8: Test evaluation. Only after all 120 runs were complete, and after the code confirmed that the protocol had not changed, I evaluated each saved model once on the official test set.

2.6 Data Analysis

For each training size, I compared each dropout rate with the no-dropout model using the seed pairing described in Section 2.1. For each seed, I calculated the difference between the dropout model and its no-dropout partner (dropout minus no dropout). Across the 10 seeds, I then reported:

- the mean difference,
- a 95% confidence interval for the mean difference, based on the t-distribution with 9 degrees of freedom,
- a two-sided paired t-test p-value,<sup>16</sup> and
- the number of seeds (out of 10) in which the difference was positive.

Because I made 8 comparisons for each measure (2 dropout rates × 4 training sizes), testing many comparisons increases the chance of a false positive. To account for this, I adjusted the p-values within each measure using the Holm method,<sup>19</sup> and I treated adjusted p-values below 0.05 as statistically significant.

To test whether the effect of dropout differed between 2,000 and 10,000 training images (Hypotheses 3 and 4), I compared the paired differences at the two sizes using Welch's t-test,<sup>20</sup> which does not assume the two groups have equal variance.

All analyses that were planned in the protocol are reported, including those that were not statistically significant. After examining the results, I also performed two additional analyses of overfitting that were not part of the original plan: the generalization gap at a matched point in training (epoch 20, which every run reached), and the increase in validation loss between each model's best epoch and the end of its training. These were compared with the same paired method and Holm correction. Because these analyses were added after seeing the results, I report them separately as exploratory.

The analysis was carried out in Python using pandas 3.0.5, NumPy 2.5.3, scikit-learn 1.9.1 (for F1 scores and confusion matrices), SciPy 1.18.1 (for the t-tests and confidence intervals), and Matplotlib 3.11.2 (for figures).

2.7 Ethical Considerations

This study did not involve human participants, so informed consent was not required. Fashion-MNIST is a publicly available dataset of product images and contains no personal information. To support transparency and reproducibility, I saved all code, the fixed data split, the frozen protocol, and every raw result, and I reported all planned comparisons regardless of outcome. The use of AI-assisted tools is disclosed in the Acknowledgments.
