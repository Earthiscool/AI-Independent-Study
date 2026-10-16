2.1 Dataset and Data Preparation

For this experiment, I used the Fashion-MNIST dataset introduced by Xiao et al. (2017). Fashion-MNIST contains grayscale images of clothing items that are 28 × 28 pixels in size. The dataset contains ten different clothing categories, including T-shirts, trousers, pullovers, dresses, coats, sandals, shirts, sneakers, bags, and ankle boots.

I chose Fashion-MNIST because it contains several clothing categories that look similar to each other, which makes it useful for investigating the types of mistakes a neural network makes. In particular, I wanted to focus on shirts, T-shirts, pullovers, and coats because these categories share several visual features and can be difficult for a neural network to distinguish.

Fashion-MNIST contains 60,000 training images and 10,000 official test images. For this experiment, I used the original training portion of the dataset to create my training and validation sets. I reserved 5,000 images for validation and used the same validation images throughout the entire experiment. I did not use the official Fashion-MNIST test set because I wanted to keep it separate for the final evaluation of my research.

I converted the images into PyTorch tensors using transforms.ToTensor(), which also scales their pixel values between 0 and 1. I used a fixed random seed of 42 to create the original training and validation splits during Week 1 of my independent study. I saved these indices so that I could reuse the exact same images throughout my research.

For the training data, I created two different subsets containing 2,000 and 10,000 images. The smaller subset was created using the first 2,000 images from the same fixed 10,000-image training subset. This means that all 2,000 images were also included in the larger training subset. I did this to make the comparison more controlled because the main difference between the two datasets was the number of additional training examples.

I also checked that none of the training images overlapped with the validation images. This was important because I wanted to measure how well the neural network could classify images that it had not already learned from during training.

2.2 Neural Network Architecture

For this experiment, I used the multilayer perceptron (MLP) that I originally built during Week 2 of my independent study. I decided to use the same model architecture rather than building a completely different neural network because I wanted to focus specifically on how dropout affects classification performance.

The MLP contained an input layer, two hidden layers, and an output layer. Each Fashion-MNIST image was first flattened from a 28 × 28 image into a vector containing 784 pixel values. These values were then passed into the first hidden layer, which contained 128 neurons. The second hidden layer contained 64 neurons, and the final output layer contained 10 neurons, one for each clothing category.

I used the ReLU activation function after both hidden layers. ReLU introduces nonlinearity into the neural network, which allows it to learn more complicated patterns instead of only learning linear relationships between the input pixels.

The final output layer produced ten scores, also called logits, which represented the model's predictions for each clothing category. I used these scores with the cross-entropy loss function to measure how incorrect the model's predictions were during training.

2.3 Dropout Implementation

To investigate the effect of dropout, I created two versions of the same MLP. The first version was the baseline model, which did not include dropout. The second version used dropout with a probability of 0.3 after each hidden layer's ReLU activation function.

Dropout randomly sets some of the hidden activations to zero during training. With a dropout probability of 0.3, each activation has a 30% chance of being turned off during a training forward pass. The purpose of this technique is to prevent the neural network from becoming too dependent on specific features or combinations of neurons.

I applied dropout after both hidden layers because I wanted to investigate how regularizing the network's learned features would affect its classification accuracy and mistakes.

Both models used the same number of layers, the same number of neurons, and the same activation functions. The only difference was whether dropout was enabled. Dropout was active during training but disabled when calculating validation results.

Keeping the architectures the same was important because I wanted to make sure that any differences in performance could be compared based on the use of dropout rather than changes to the structure of the neural network.

2.4 Learning Rate Pilot Experiment

Before conducting the final experiments, I ran a small pilot experiment to determine which learning rate I should use.

The learning rate controls how much the neural network's weights are adjusted during training. If the learning rate is too small, the model might learn very slowly. However, if it is too large, the model might have difficulty finding weights that reduce the loss consistently.

For the pilot experiment, I tested three different learning rates: 0.01, 0.05, and 0.1. I used the baseline MLP without dropout and trained it for two epochs using both the 2,000-image and 10,000-image training subsets. I used the same random seed of 42 for these pilot runs.

After each pilot run, I recorded the model's validation accuracy and macro-F1 score. I compared the average validation accuracy across the two dataset sizes for each learning rate.

Based on these results, I selected a learning rate of 0.1 because it produced the highest average validation accuracy among the three values tested.

Once I selected the learning rate, I kept it fixed throughout all 12 final experiments. This helped make the comparison between the baseline and dropout models more consistent.

2.5 Training Procedure

I trained both versions of the MLP using two different training dataset sizes: 2,000 images and 10,000 images.

For each dataset size, I trained the baseline model without dropout and the model with dropout set to 0.3. I repeated each condition using three different random seeds: 42, 123, and 456. This gave me four experimental conditions and a total of 12 training runs.

I used multiple random seeds because neural networks begin with randomly initialized weights, and these weights can affect how the model learns. By repeating each experiment three times, I could compare whether the results were consistent rather than relying on one individual training run.

To keep the experiments fair, I used the same batch size of 32, the same learning rate of 0.1, and the same stochastic gradient descent (SGD) optimizer for every model. Each model was trained for 10 epochs.

During each epoch, the training data was divided into batches of 32 images. The model made predictions for each batch, and I calculated the cross-entropy loss by comparing the predictions with the correct labels.

I then used backpropagation to calculate the gradients of the model's trainable weights. The SGD optimizer used these gradients to update the weights so that the model could gradually improve its predictions.

After every epoch, I evaluated the model using the fixed validation dataset of 5,000 images. I recorded the training loss, training accuracy, validation loss, and validation accuracy for each epoch.

I also recorded the time required to complete each training run. All experiments were performed using PyTorch, with Apple's MPS device used to accelerate the computations.

2.6 Evaluation Metrics

To evaluate the performance of the models, I used several different metrics rather than only comparing their overall accuracy.

The first metric was validation accuracy, which measures the percentage of validation images that the neural network classified correctly. I used this as my primary evaluation metric because it provides a general measurement of how well each model performs on unseen images.

However, overall accuracy does not explain what types of mistakes the neural network is making. Two models could achieve similar accuracy while incorrectly classifying different clothing categories.

Because of this, I also calculated precision, recall, and F1 scores for the clothing categories. Precision measures how often a prediction for a particular category is correct, while recall measures how many of the actual examples of that category the model correctly identifies.

The F1 score combines precision and recall into one measurement. I also calculated macro-F1, which takes the average F1 score across all ten categories and gives each category equal importance.

I wanted to include these metrics because a model might perform well overall but still struggle with certain types of clothing.

2.7 Analysis of Similar Clothing Categories

One of the main goals of my research was to investigate whether dropout changes the mistakes a neural network makes between visually similar clothing categories.

For this part of the experiment, I focused specifically on T-shirts, pullovers, coats, and shirts. I chose these four categories because they have similar shapes and visual features, which makes them more difficult to distinguish compared with categories such as shoes and bags.

I used confusion matrices to examine how often the model predicted one clothing category when the correct answer was another. A confusion matrix shows the number of correct and incorrect predictions for each category, allowing me to identify which categories were most commonly confused with one another.

I also calculated a within-group confusion rate for the four selected clothing categories. For this measurement, I counted the number of times an image belonging to one of these four categories was incorrectly predicted as a different category within the same group. I then divided this number by the total number of validation images belonging to those four categories and multiplied by 100 to express it as a percentage.

This measurement specifically focuses on confusion between similar clothing categories. It does not count mistakes where one of these categories is incorrectly classified as a category outside the selected group.

By comparing the within-group confusion rates for the baseline and dropout models, I could investigate whether adding dropout helped the neural network distinguish between visually similar clothing items.

I compared these results separately for the 2,000-image and 10,000-image training conditions to determine whether the effect of dropout changed when more training data was available.

2.8 Comparing Results Across Experiments

After completing all 12 training runs, I organized the results into CSV files so that I could compare the different experimental conditions.

For each condition, I calculated the average validation accuracy, average macro-F1 score, and average within-group confusion rate across the three random seeds. I also calculated the standard deviation of validation accuracy to examine how much the results varied between different runs.

I created graphs to compare the validation accuracy of the baseline and dropout models at both dataset sizes. I also plotted the training and validation losses across the ten epochs to examine how the models learned throughout training.

Finally, I compared confusion matrices for the four similar clothing categories to identify which classification mistakes were most common and whether those patterns changed when dropout was introduced.

I used these comparisons to investigate my original research question: whether dropout changes both the overall accuracy of a neural network and the types of mistakes it makes when trained on different amounts of data.

By keeping the model architecture and training settings consistent, I attempted to isolate the effects of dropout and training dataset size while making the experiment as reproducible as possible.