# Deep Learning for Coders / Chapter-5 \(part-2\) / Week-7

**Published:** July 21, 2021

**Making a Model Better**

**Contents:**

1. Source of the blogpost
2. A Quick recap from Week-6
3. Keypoints
   1. How to interpret a model
   2. Learning Rate Finder
   3. Unfreezing & Transfer Learning
4. Conclusion
5. Credits

**Source of the blogpost:**

* This blogpost is based on Chapter-5 from the Fastbook \[“DeepLearning for Coders with FastAI & PyTorch”\]. And this is created after my participation in Week-7 of FastBook reading session organized by _Weight & Biases_, hosted by _AmanArora_.

**A Quick recap from Week-6:**

* In week-6 we have covered the first half of the chapter-5 which is PetBreeds and learned several key concepts like Presizing / Augmentation of Images, Datablocks and Cross-Entropy loss.
* The main idea behind augmenting the images is to reduce the number of computations and lossy operations. This will also more efficient processing on the GPU.
* A `datablock` is a generic container to quickly build ‘Datasets’ and ‘DataLoaders’ .
* Cross-Entropy loss is a combination of using the negative log likelihood on the log values of the probabilities from the softmax function.

> Link to the Week-6 blogpost: [Deep Learning for Coders / Chapter-5 / Week-6 - Ravi Chandra Veeramachaneni](https://ravichandraveeramachaneni.github.io/posts/bp8/)

**Keypoints:**

* The following are the key points from 2nd half of the PetBreeds chapter.

**How to interpret a model:**

* A usual way of interpreting or evaluating the model is looking at the metrics like a confusion matrix which will show where the model is performing poorly.
* But one of the toughest part of interpreting confusion matrix is when it has multi-category.
* We can overcome this by using a FastAI convenience function like `most_confused`

![Screen Shot 2021-07-28 at 1 19 13 PM](https://user-images.githubusercontent.com/14807933/127401027-c043cd96-142b-4064-9e3a-cdca32e6be0f.png)

**Learning Rate Finder:**

* One of the key points to consider when training a model would be to have a right learning rate and that can be found using the `lr_find` method, originally proposed by researcher Leslie Smith.

  ```text
  learn = cnn_learner(dls, resnet34, metrics=error_rate)
  lr_min,lr_steep = learn.lr_find()
  ```

![Screen Shot 2021-07-28 at 1 38 47 PM](https://user-images.githubusercontent.com/14807933/127401071-ff4c7395-0cb5-42ea-b6dc-0e78acedf1ba.png)

**Unfreezing & Transfer Learning:**

* When training the model on a certain task, the optimizer should update the weights in the randomly added final layers. But we do not change the weights in the rest of the neural network at all. This is called _freezing_ the pre-trained layers.
* When we are fine tuning the model, the fastai does two things:
  * Trains the randomly added layers for one epoch, with all other layers frozen.
  * Unfreezes all of the layers, and trains them all for the number of epochs requested.
* Instead of doing the fine-tuning from the library, we will also be able to do that manually. In that case we can unfreeze the layers by using the below code snippet.

**Conclusion:**

* To make our model better we can perform many steps right from data preparation which involves techniques like Pre-sizing to fine\_tuning the model by fining proper learning rates. So each step has to be taken care in the whole process to yield better accuracy.

**Credits:**

* All the images used are from the fast book.

