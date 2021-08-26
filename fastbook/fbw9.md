# Deep Learning for Coders / Chapter-7 / Week-9

**Published:** August 04, 2021

**Techniques to Train a State-Of-The-Art Model**

**Contents:**

1. Source of the blogpost
2. Re-cap of Chapter-6
3. Introduction
4. Imagenette
5. Normalization
6. Progressive Resizing
7. Test time augmentation
8. Mixup
9. Label Smoothing
10. Conclusion
11. Credits

**Source of the blogpost:**

This blogpost is based on Chapter-7 from the Fastbook \[“DeepLearning for Coders with FastAI & PyTorch”\]. And this is created after my participation in Week-9 of FastBook reading session organized by _Weight & Biases_, hosted by _AmanArora_.

**Re-cap of Chapter-6:**

In the previous chapter we have understood several topics such as Multilabel classification, one-hot encoding binary cross-entropy loss, Regression etc. Multi-label classification refers to the problem of identifying the categories of objects in images that may not contain exactly one type of object. There may be more than one kind of object, or there may be no objects at all in the classes that you are looking for. In the use case above we represent data by having 1’s in the vector of 0’s. This technique is called One-hot encoding.

So, in the above case where is there are more than a single label, the cross-entropy doesn’t hold good anymore. The Cross-Entropy loss is a combination of using the negative log likelihood on the log values of the probabilities from the softmax function. But in the case of multi-category labels, we don’t have the probabilities rather we have the one-hot encoded values. In this case, the best option would be the binary cross-entropy loss which is basically just mnist\_loss along with log. Also in this case we need to have a special performance metric called `accuracy_multi`.

A regression problem is when the output variable is a real or continuous value, such as “salary” or “weight”. Many different models can be used, the simplest is the linear regression. It tries to fit data with the best hyper-plane which goes through the points. In the case of the regression problem, the loss that can used is MSELoss \(Mean Squared Error loss\). The MSE tells you how close a regression line is to a set of points.

**Introduction:**

> It is better to fail fast than very late. And it is always better to run more experiments on a smaller dataset rather running a single experiment on a large dataset.

This chapter introduces a new dataset called \(Imagenette\)[https://github.com/fastai/imagenette](https://github.com/fastai/imagenette). Imagenette is a subset of the original Imagenet dataset but has only 10 categories of classes which are very different. This dataset has full-size, full-color images, which are photos of objects of different sizes, in different orientations, in different lighting, and so forth.

**Imagenette:**

> Important message: the dataset you get given is not necessarily the dataset you want.

This dataset has been created by fast.ai team to quickly experiment with the ideas and to give the opportunity to iterate quickly. Lets see how can we work with this dataset and then apply techniques which can be used on larger datasets like Imagenet as well.

**Step 1:** Downloading dataset & building The Datablock

```text
path = untar_data(URLs.IMAGENETTE)
dblock = DataBlock(blocks=(ImageBlock(), CategoryBlock()),
                   get_items=get_image_files,
                   get_y=parent_label,
                   item_tfms=Resize(460),
                   batch_tfms=aug_transforms(size=224, min_scale=0.75))
dls = dblock.dataloaders(path, bs=64) 
```

**Step 2:** Creating a Baseline & Training the model

```text
model = xresnet50(n_out=dls.c)
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
learn.fit_one_cycle(5, 3e-3)
```

![1](https://user-images.githubusercontent.com/14807933/129126163-6d0a63d7-fb07-4e6f-ab75-038ec9d6470d.png)

So far we have achieved about 83.3% of accuracy. Let’s try to apply some techniques that would improve the performance.

**Normalization:**

> One of the strategy in the data pre-processing that will help a model perform better is to normalize the data.

Data which has mean of 0 and a standard deviation of 1 is referred as Normalized data. But most of the data like images used is in between 0 to 255 pixels or between 0 & 1. So, we do not have the normalized data in either case. So to normalize the data, in fastAI we can pass `Normalize` transform. This transform will take the mean and standard deviation we want and transform the data accordingly. Normalization is an important technique that can be used when using pre-trained models.

**Note:** When using the cnn\_learner with a pre-trained problem, we need not add the `Normalize` transform since the fastAI library automatically adds it.

**Step 3:** Adding Normalization

```text
def get_dls(bs, size):
    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   get_items=get_image_files,
                   get_y=parent_label,
                   item_tfms=Resize(460),
                   batch_tfms=[*aug_transforms(size=size, min_scale=0.75),Normalize.from_stats(*imagenet_stats)])
    return dblock.dataloaders(path, bs=bs)
```

**Step 4:** Training the model again with normalization added

After normalization, we have achieved an accuracy of 82.2%. Not an huge improvement from the previous, but for some strange reason there is a slight drop in performnace.

![2](https://user-images.githubusercontent.com/14807933/129126220-db79ad04-6a64-423c-86d1-83c426d12d8d.png)

The other technique we can employ here for training is ti start small and then increase as required. All the above steps are confined to train images which are at size 224. So, we can start with much smaller size and increase it and this technique is called Progressive Resizing.

**Progressive Resizing:**

> Progressive resizing: Gradually using larger and larger images as you train.

Spending most of the epochs training with small images, helps training complete much faster. Completing training using large images makes the final accuracy much higher. In the process, since we will be using different size of images, we can use `fine_tune` to tune the model. And if we closely observed this is kind of a data augmentation technique.

**Step 5:** Create a data loader and try to fit into the model.

```text
dls = get_dls(128, 128)
learn = Learner(dls, xresnet50(n_out=dls.c), loss_func=CrossEntropyLossFlat(), 
                metrics=accuracy)
learn.fit_one_cycle(4, 3e-3)
```

![3](https://user-images.githubusercontent.com/14807933/129126270-4793c5ec-bbb4-4f14-8bc4-454f3e67f652.png)

**Step 6:** Replace the data loader and fine\_tune it.

```text
learn.dls = get_dls(64, 224)
learn.fine_tune(5, 1e-3)
```

![4](https://user-images.githubusercontent.com/14807933/129126301-3d1127cd-48a9-4d48-a1e5-b06e4e426105.png)

From the above step it is evident that the Progressive resizing has achieved a good improvement in the accuracy of about 86.3%. However, it’s important to understand that the size of the Image at maximum could be the size of the image available on disk. Also, another caution on the resizing part is to not damage the pertained weights. This might happen if we have the the pre-trained weights similar to the weights in the transfer learning. The next technique to apply to the model is to apply data augmentation to the test set.

**Test time Augmentation:**

> Test Time Augmentation \(TTA\): During inference or validation, creating multiple versions of each image, using data augmentation, and then taking the average or maximum of the predictions for each augmented version of the image.

Traditionally we use to perform training data augmentation with different techniques. When it comes to validation set, the fastAI for instance applies the center cropping. Center cropping is useful in some use cases but not all. This is because cropping from center may entirely discard any images on the borders. Instead on way would be to stretch and squish instead of cropping. However this becomes a hectic problem for model to learn those new patterns. Another way would be to select a number of areas to crop from the original rectangular image, pass each of them through our model, and take the maximum or average of the predictions. We could do this around different values across all of our test time augmentation parameters. This is known as /test time augmentation/ \(TTA\).

**Step 7:** Trying TTA

```text
preds,targs = learn.tta()
accuracy(preds, targs).item() 
```

![5](https://user-images.githubusercontent.com/14807933/129126337-75b3dff7-7d0c-4f4e-a687-b77af033d3f3.png)

We can see that the above technique has turned out well on improving accuracy to about 87.5%. However the above process slows down the inference by number of times we are averaging for TTA. So we can try another technique called Mixup.

**Mixup:**

Mixup is a very powerful data augmentation technique that can provide dramatically higher accuracy, especially when you don’t have much data and don’t have a pre-trained model that was trained on data similar to your dataset. Mixup technique talks about the data augmentation for the specific kind of dataset and fine tuned as needed. Mixup works as follows, for each image:

1. Picking a random image from your dataset.
2. Picking a weight at random.
3. Taking a weighted average \(from step 2\) of the selected image with your image; this will be your independent variable.
4. Taking a weighted average \(with the same weight\) of this image’s labels with your image’s labels; this will be your dependent variable.

> Note: For mixup, the targets need to be one-hot encoded.

One of the reasons that Mixup is so exciting is that it can be applied to types of data other than photos. But, the issue with this technique might be the labels getting bigger than 0 or smaller than one as opposed to the one-hot encodings. So, we can handle this through label smoothing.

**Label Smoothing:**

In “Classification Problems”, our targets are one-hot encoded, which means we have the the model return either 0 or 1. Even a smallest of the difference like 0.999 will encourage the model to overfit and at inference the model that is not going to give meaningful probabilities. Instead to avoid this we could replace all our 1s with a number a bit less than 1, and our 0s by a number a bit more than 0, and then train. This is called label smoothing. And this will make the model generalize better.

**Conclusion:**

All the techniques described above are kind of eye opening on how we can build techniques that could augment each other and sometimes better than others. All these techniques will be applied to a real dataset and results will be published soon with description.

**Credits:**

* All the images are from [my colab](https://gist.github.com/RaviChandraVeeramachaneni/e6b62ec22dc464d569d3b1ccf9f28d5c) that I used for experimenting with all these above features.
* All the code is from [fastbook](https://colab.research.google.com/github/fastai/fastbook/blob/master/07_sizing_and_tta.ipynb#scrollTo=QvafYrrhYx1T)

