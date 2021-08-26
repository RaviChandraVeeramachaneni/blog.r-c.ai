# Deep Learning for Coders / Chapter-5 / Week-6

**Published:** July 14, 2021

KeyPoints from Chapter-5 \(PET BREEDS\)

This blogpost is based on Chapter-5 from the Fastbook \[“DeepLearning for Coders with FastAI & PyTorch”\]. Some of the key topics in the following blogpost include:

1. Presizing / Augmentation of Images.
2. Datablock.
3. Cross-Entropy loss.

_Presizing / Augmentation of Images:_

* The main idea behind augmenting the images is to reduce the number of computations and lossy operations. This also results in more efficient processing on the GPU.
* To make the above possible we need our images to have same dimensions, so they can be easily collated.
* Some of the challenges in doing the augmentation is that when we resize, the data could be degraded, new empty zones are introduced etc.

![Image\_Augmentation](https://user-images.githubusercontent.com/14807933/126576976-f2524a61-273e-4985-be73-e37f6316db7c.png) ![Image\_augmentation1](https://user-images.githubusercontent.com/14807933/126576986-dacecd78-46cd-4649-a05b-f1ada135e5b8.png)

* So how can we over come these?

There are around two strategies:

1. Resize images to relatively larger dimensions than the target training dimensions.
2. Having all the augmentation operations done at once on the GPU at end of processing rather than performing operations individually and interpolating multiple times.
   * Two important things to note in the below example:

     ```text
     pets = DataBlock(blocks = (ImageBlock, CategoryBlock),
                 get_items=get_image_files, 
                 splitter=RandomSplitter(seed=42),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                 *item_tfms=Resize(460)*,
                 *batch_tfms=aug_transforms(size=224,* min_scale=0.75))
     ```
3. `item_tmfs` is applied to each individual image before its copied to GPU. And it ensures three things, that all images are the same size and on the training set, the crop area is chosen randomly and the validation set, the center square of the image is chosen.
4. `batch_tfms` is applied to a batch all at once on the GPU.

_Datablock_:

* A `datablock` is a generic container to quickly build ‘Datasets’ and ‘DataLoaders’ .
* To build a datablock we need to know what kind of TransformBlock like a ImageBlock, CategoryBlock, which method to fetch the items like `get_image_files` , how to split the images, how to get the labels and any transformations to be applied.
* The below example repeated same as above in this context is a how we create a datablock:

  ```text
  pets = DataBlock(blocks = (ImageBlock, CategoryBlock),
                 get_items=get_image_files, 
                 splitter=RandomSplitter(seed=42),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                 item_tfms=Resize(460),
                 batch_tfms=aug_transforms(size=224, min_scale=0.75))
  ```

* Once we have the datablock, we can get the dataloader object by just calling the dataloaders method on that datablock

  ```text
  dls = pets.dataloaders(path/"images")
  ```

* So to ensure that we have the datablock created properly without any errors, we can do that with the following piece of code

  ```text
  dls.show_batch(nrows=1, ncols=3)
  ```

* One of the important debugging method to learn when we have trouble creating a proper datablock is summary method. The summary method will provide very detailed stack trace

  ```text
  pets.summary(path/"images")
  ```

_Cross-Entropy Loss:_

> Cross-Entropy loss is use of negative loss on probabilities. Or in simple terms Cross-Entropy loss is a combination of using the negative log likelihood on the log values of the probabilities from the softmax function.

The below image depicts the cross-entropy loss: ![Cross-Entropy\_loss](https://user-images.githubusercontent.com/14807933/126577017-21f15176-31b8-4b0a-a3be-3657701f6872.png)

_Keypoints about Cross-Entropy loss:_

* The best suited loss for the Image data and a categorical outcome is Cross-Entropy loss. When we haven’t provided the loss function we want to use, the fastAI by default will pick the cross-entropy.
* Cross-Entropy loss works even when with multi-categories of dependent variables.
* And this also results in faster and more reliable training.
* To transform the activations of our model into predictions, we use something called the softmax activation function.

_Softmax Function:_

* A softmax function will ensure that all the activations in the final layer of our classification model are between 0 and 1 and they all sum up to 1.
* It is more or less similar to sigmoid function.

![Sigmoid](https://user-images.githubusercontent.com/14807933/126577047-72aebabb-ef74-4eba-af28-ac1e8fc500ca.png)

* A _sigmoid function_ when applied to single column of activations from neural network will return a column of numbers from 0 and 1. Now we are chosing softmax function since we have multi-categories and we need activations per category.
* If we are trying to apply the softmax function for two categories, it returns the same values as sigmoid for the first column and those subtracted from 1 for the second column. \`\`\` def softmax\(x\): return exp\(x\) / exp\(x\).sum\(dim=1, keepdim=True\)

\`\`\`

> Exponential function \(exp\) is defined as e\*\*x, where e is a special number approximately equal to 2.718. It is the inverse of the natural logarithm function. Note that exp is always positive, and it increases very rapidly!

* We need exponential since it ensures that all numbers are positive and dividing by the sum ensures that they all add up to 1.
* And softmax function is better at picking one class among others, so it is ideal for training.
* The second part of the cross-entropy loss is Log Likelihood after the softmax function.

_Log Likelihood:_

* Lets consider an example of having 0.99 and 0.999 as probabilities they are very close but in terms of confidence the 0.999 is more confident than 0.99. So to transform the numbers between the negative infinity and 0 to 0 and 1.

![Log](https://user-images.githubusercontent.com/14807933/126577061-5ab8b11f-4853-4cff-85a2-f56f01dc57f9.png)

* So taking the mean of the positive or negative log of our probabilities \(depending on whether it’s the correct or incorrect class\) gives us the _negative log likelihood_ loss. In PyTorch, nll\_loss assumes that you already took the log of the softmax, so it doesn’t actually do the logarithm for you.
* The CrossEntropyLoss function from pytorch exactly does the same:

![Screen Shot 2021-07-21 at 5 40 13 PM](https://user-images.githubusercontent.com/14807933/126577078-fed05e5b-4ac9-4e75-9170-710ff17ab9aa.png)

_Conclusion_: Cross-Entropy loss is for multi-category classification and is simply the use of negative loss on probabilities.

Credits:

1. The Images used in the Data Augmentation and Cross-Entropy loss depiction are from google.
2. The sigmoid function plot and log likelihood plot are from the fastbook.

