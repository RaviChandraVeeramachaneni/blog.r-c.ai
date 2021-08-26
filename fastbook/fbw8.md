# Deep Learning for Coders / Chapter-6 / Week-8

**Published:** July 28, 2021

**Multi-Label Classification**

**Contents:**

1. Source of the blogpost
2. Skip / Read
   1. From the Previous Chapters
3. MultiLabel Classification
4. Binary Cross-Entropy Loss
5. Regression
6. Conclusion
7. Credits

**Source of the blogpost:**

* This blogpost is based on Chapter-6 from the Fastbook \[“DeepLearning for Coders with FastAI & PyTorch”\]. And this is created after my participation in Week-8 of FastBook reading session organized by _Weight & Biases_, hosted by _AmanArora_.

**Skip / Read:**

If you already have knowledge of the Deep Learning basics and you understand what is a model, Stochastic Gradient Descent, epoch, learning rate, softmax, cross entropy etc then you may skip this section and jump to next section. Or if you are looking to just have a glimpse of all the above terms, please read further in this section. This section has all the basics needed to understand the basic terms.

**From the Previous Chapters:**

From the Previous chapters we have learned couple of different things, right from what is a model to how to find better learning rates etc. Lets have a small recap on each thing we learned so far:

* FastAI has lots of API conveniently wrapped on top of the PyTorch for ease of use.
* In FastAI for Images, we have functions starting with Images like ImageDataLoaders and for text we have functions starting with Text like TextDataLoaders.
* In FastAI we have 2 types of transforms, Item transforms\(item\_tfms\), and the other is Batch transforms\(batch\_tfms\). The item tranformation operates on each item / input image to resize them to a similar size and the batch transform operates on the batches of items and pass them to the GPU\(s\) for training.
* And “EPOCH” is a one pass through, of all the images in training. And the process of the model learning in each epoch is called “Model Fitting”.
* Using a pre-trained dataset like IMAGENET for classifying a different task. A IMAGENET is an original dataset with 1M images used for vision tasks is called Transfer learning.
* Fine Tuning - Training the model on a general dataset & then training it on our own dataset. This is where we are using the pre-trained weights for all the layers unaltered except for the last layer\(head\). The process of retaining the model stem\(pre-trained weights\) and just training the new head is called Fine Tuning. And it’s a Transfer learning technique.
* Stochastic Gradient Descent is the way of automatically updating the weights of the neural network input’s, based on the previous result to gain the maximum performance or in simple terms better accuracy.
* Cross-Entropy loss is a combination of using the negative log likelihood on the log values of the probabilities from the softmax function.
* A softmax function will ensure that all the activations in the final layer of our classification model are between 0 and 1 and they all sum up to 1.
* One of the key points to consider when training a model would be to have a right learning rate and that can be found using the `lr_find` method, originally proposed by researcher Leslie Smith.

**MultiLabel Classification:**

Multi-label classification refers to the problem of identifying the categories of objects in images that may not contain exactly one type of object. There may be more than one kind of object, or there may be no objects at all in the classes that you are looking for. See two examples below where we have a bear dataset with a dog included named bear and another example where the cat is classified as cat and horse.

![Screen Shot 2021-08-04 at 2 03 53 PM](https://user-images.githubusercontent.com/14807933/128271515-83be0a0e-39e9-4443-9316-2facdb863897.png)

![Screen Shot 2021-08-04 at 2 08 35 PM](https://user-images.githubusercontent.com/14807933/128271527-4f7c2d57-7829-471e-81b9-61d2e2cae2e5.png)

As a note, in FastAI we can handle the multi-labels with `MultiCategoryBlock` which will encode all the vocabulary into a list of 0’s and have 1s where data is present. So, by checking where 1’s are we can identify which category\(s\) that image belongs to. This technique of representing the data in 1’s on a vector of 0s is called One-hot encoding.

**Binary Cross-Entropy Loss:**

In the case of single category labels, we have the cross-entropy loss. The Cross-Entropy loss is a combination of using the negative log likelihood on the log values of the probabilities from the softmax function. But in the case of multi-category labels, we don’t have the probabilities rather we have the one-hot encoded values. In this case, the best option would be the binary cross-entropy loss which is basically just mnist\_loss along with log.

```text
def binary_cross_entropy(inputs, targets):
    inputs = inputs.sigmoid()
    return -torch.where(targets==1, 1-inputs, inputs).log().mean()
```

So, once we are ready with the data and preparing to create Learner for training, we do not need to explicitly provide the loss. The FastAI will pick up the binary corss-entorpy loss by default.

Now that we have the loss ready, we need to pick a metric which is accuracy by default for all the classification problems we worked on. But the accuracy is not a good fit for this problem of multi-label since for each image we could have more than one prediction. So we need to use `accuracy_multi` with a threshold that will address the problem.

> Note: Since the threshold for the accuracy\_multi is by default 0.5, we can override the function using the `partial` function from python.

Example of the Learner in this case:

```text
learn = cnn_learner(dls, resnet50, metrics=partial(accuracy_multi, thresh=0.2))
learn.fine_tune(3, base_lr=3e-3, freeze_epochs=4)
```

**Regression:**

A regression problem is when the output variable is a real or continuous value, such as “salary” or “weight”. Many different models can be used, the simplest is the linear regression. It tries to fit data with the best hyper-plane which goes through the points.

![Screen Shot 2021-08-04 at 2 08 35 PM](https://user-images.githubusercontent.com/14807933/128271611-31944d0a-5dd2-4470-9221-67afed540290.png)

An image regression problem refers to learning from a dataset where the independent variable is an image, and the dependent variable is one or more floats. And image regression is simply a CNN under the hood. One of the key perspective to consider while building a datablock for regression is to use `pointblock` instead of a category block since the labels represents coordinates. Another important point to remember while construction the Learner is to provide the `y_range=(-1,1)` attribute to make sure that we give the range of the rescaled coordinates.

```text
learn = cnn_learner(dls, resnet18, y_range=(-1,1))
```

In the case of the regression problem, the loss that can used is MSELoss \(Mean Squared Error loss\). The MSE tells you how close a regression line is to a set of points. It does this by taking the distances from the points to the regression line \(these distances are the “errors”\) and squaring them.

**Conclusion:**

All the problems like single-label classification, multi-label classification & regression seems to work on basis of same model except for the loss function that changes every time. So, we need to keep an eye on hyper parameters and loss which will effect the results.

**Credits:**

All the images are picked from google for visual understanding as needed in the blogpost and modified in above cases like Multi-label classification.

