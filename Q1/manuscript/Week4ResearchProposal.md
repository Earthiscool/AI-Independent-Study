Research Proposal

Title: Beyond Overall Accuracy: How Dropout Affects Classification Errors with Limited Training Data

Research Question

How does dropout affect the overall accuracy of a neural network and the mistakes it makes when identifying similar clothing categories in Fashion-MNIST, especially when the model is trained on 2,000 images compared to 10,000 images?

Background and Motivation

Dropout is a regularization technique that randomly turns off some neurons during training. This helps prevent the neural network from becoming too dependent on certain features and can reduce overfitting. Srivastava et al. (2014) found that the effectiveness of dropout can change depending on how much training data is available.

Fashion-MNIST is a dataset introduced by Xiao et al. (2017) that contains images of different types of clothing. Previous research has shown that neural networks often have difficulty distinguishing between similar clothing categories, such as shirts, T-shirts, pullovers, and coats (Nocentini et al., 2022; Ahmad et al., 2026).

However, I want to investigate whether adding dropout changes the types of mistakes a neural network makes when it has different amounts of training data. Instead of only looking at the model's overall accuracy, I also want to understand whether dropout helps the model distinguish between clothing categories that look similar.

Hypothesis

I predict that dropout will reduce the number of mistakes between similar clothing categories more effectively when the model is trained on 10,000 images rather than 2,000 images. My reasoning is that when the model has very little training data, dropout might make it harder to learn important patterns because some neurons are randomly turned off during training. With more training examples, the model may be able to learn these patterns more effectively while still benefiting from dropout.

Methodology

For this experiment, I will use Fashion-MNIST and the multilayer perceptron (MLP) that I built during Weeks 1 and 2. My neural network has two hidden layers containing 128 and 64 neurons, with ReLU activation functions between the layers.

I will compare two versions of this neural network. The first will be the original model without dropout, and the second will include dropout with a probability of 0.3 after each hidden layer's activation function.

I will train both models using two different dataset sizes: 2,000 and 10,000 images. Each experiment will be repeated using three different random seeds, giving me 12 total training runs. To make the comparison fair, I will keep the batch size at 32, use the same optimizer and learning rate, and initially train each model for 10 epochs. I will also use the same 5,000 validation images for every experiment.

Before starting the final experiments, I will use a small pilot experiment to select a learning rate. I will not use the official Fashion-MNIST test set until the final evaluation.

Evaluation

To evaluate my results, I will first compare the overall validation accuracy of each model. I will also calculate precision, recall, and macro-F1 scores to get a better understanding of how well the models classify different clothing categories.

I will pay particular attention to shirts, T-shirts, pullovers, and coats because these categories are visually similar. I will use confusion matrices to examine how often the models incorrectly classify one of these categories as another.

I will also compare the results across the three random seeds and examine the training and validation loss graphs to understand how dropout affects learning.

Expected Contribution and Limitations

Through this research, I hope to better understand whether dropout changes the kinds of mistakes a neural network makes, rather than only improving its overall accuracy. I also want to see whether these effects change when the model has access to different amounts of training data.

It is possible that dropout will not consistently improve the results, but that would still help me understand its limitations.

One limitation is that I am only testing one neural network architecture, two dataset sizes, and one dropout probability. Because of this, my results will not necessarily apply to every neural network or dataset. The experiment will also use only three random seeds per condition, so I will need to be careful about drawing broad conclusions.