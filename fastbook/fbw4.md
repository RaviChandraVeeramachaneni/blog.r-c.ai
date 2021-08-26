# Deep Learning for Coders / Chapter-4\(Part-1\) / Week-4

**Published:** June 30, 2021

**KeyPoints from Chapter-4 1st Half \(Training a Digit Classifier\)**

We are first going to read the summary of the first half of the chapter in simple terms and learn in depth about each topic & jargons in the detailed explanation provided next which includes examples as needed.

**Summary:** In this chapter we are going to recognize Hand written digits. For the purpose we are going to use [MNIST](https://en.wikipedia.org/wiki/MNIST_database) database. So for that we need images in form of numbers. A image is an array of numbers / pixels under the hood. We need this because a computer only understand numbers. So, we try to convert all images into numbers and hold them in a data structure like Numpy Array or PyTorch tensor which will make the computations easier. Each Image is two dimensional in nature since they are grey scaled. All the images are now stacked to create a single tensor of 3 dimensions which will help to calculate the mean of each pixel. Now we do this mean calculation to achieve a single image of 3 or 7, which our model learns as an ideal 3 or 7. Now given an image of 3 or 7 whether from a training set or validation set we can use distance techniques like L1 norm or L2 norm to compute the distance between that image and the ideal 3 or 7 image. So the accuracy is the metric we will consider for knowing how good our model is performing.

\*\*Above terms simplied: \*\*

1. An image under the hood, in a computer is an array of numbers.
2. It can be represented using numpy arrays or pytorch tensors.

```text
Example:
Using Numpy Arrays:
array(img) - To get full array 
array(img)[i:j, i:j] - To get the partial array we can provide rows, cols

Using PyTorch tensors:
tensor(img) - To get full tensor
tensor(img3)[i:j, i:j] - To get the partial tensor we can provide rows, cos
```

1. Both Numpy Array & PyTorch Tensor are same except for the naming. An Numpy array is a simple 2-dimensional representation of the image where as the PyTorch tensor is a multi-dimensional representation of the image. The below image shows the example of an image illustrated as an array & tensor and its dimensions.

Note: Since the image is a simple grey image \(No Color\), we have just 2 dimensions. If it’s an color image than we would have 3 dimensions \(R,G,B\)

![Screen Shot 2021-07-06 at 6 39 43 PM](https://user-images.githubusercontent.com/14807933/124732694-abb10100-ded0-11eb-9249-bd150c7df975.png)

1. Each number in the image array is called a Pixel & each pixel is in between 0 to 255.
2. The MNIST images are 28 \* 28 \(total image size 784 pixels\).
3. A Baseline is a simple model which will perform reasonably well. So it is always better to start with a baseline and keep improving the model to see how the new ideas improve the model performance/accuracy.
4. Rank of a Tensor: A Rank of a tensor is simply the number of dimensions or axes. For instance, a three dimensional tensor can also be referred as rank-3 tensor.
5. Shape of Tensor: Shape is the size of each axis of the tensor.
6. To get an ideal image of 3 or 7, we need to perform the below steps: \(Each Function shown here is explained in detailed with examples in the next section\)
7. Convert all the images into tensors using tensor\(Image.open\(each\_image\)\).
8. Wrap the converted images into a list of image tensor’s.
9. Stack all the list of image tensors so that it creates a single tensor of rank-3 using torch.stack\(list\_of\_image\_tensors\). In simple terms a 3-dimensional tensor.
10. If needed we have to cast all of the values to float for calculating mean by using float\(\) function from pytorch library.
11. Final step is to take the mean of the image tensors along dimension-0 for the above stacked rank-3 tensor using mean\(\) function. For every pixel position, the mean\(\) will compute the average of that pixel over all the images.
12. The result, is the tensor of an ideal 3 or 7 calculated by computing the mean of all the images which will have 2-dimensions like our original images.
13. To classify if an Image is a 3 or 7, we can use a simple technique like finding the distance of that image from an ideal 3 or ideal 7 computed in the earlier steps.
14. Distance calculation can be done using either L1 normalization or L2 normalization. A simple distance measure like adding up differences between the pixels would not yield accurate results due to positive & negative values \(Note: Why Negative Values ? Remember we have lot of 0’s in the image and other image may contain a value at that same pixel which would result in a negative value\).
15. L1 norm: Taking the mean of the absolute value of the differences between 2 image tensors. A absolute value abs\(\) is a function that replaces negative values with positive values. This is also referred as Mean Absolute difference. This can be calculated using F.l1\_loss\(any\_3.float\(\), ideal\_3\).
16. L2 norm: Taking the mean of the square of the differences and then take a square root. So Squaring a difference will make the value positive & then square root cancels the square effect. This is also referred to as Root Mean Squared Error \(RMSE\). This can be calculated using F.mse\_loss\(any\_3.float\(\), ideal\_3\).
17. The result of the above computation would yield the loss. Higher the loss, lesser the confidence of that image being 3 vs 7.
18. In practice, we use accuracy as the metric for our classification models. It is computed over the training data to make sure overfitting occurs.
19. Broadcasting is a technique which will automatically expand the tensor with the smaller rank to have same sizes as one with the larger rank.

```text
tensor([1,2,3]) + tensor(1)

Output: tensor([2, 3, 4])
```

1. In broadcasting technique the PyTorch never creates copies of lower ranked tensor.

**Some of the new things learned from fastAI library:**

1. ls\(\) → L - A function that list the count of items & contents of the directory. It returns a fastai class called L which is similar to python built-in List class with additional features.
2. Image.open\(\) → PngImageFile: A class from Python Image Library \(PIL\) used for operations on images like viewing, opening & manipulation.

```text
image_to_display = Image.open(path_to_image)
```

1. A Pandas.DataFrame\(image\) → DataFrame: A function that takes an image, converts that into a DataFrame Object & returns it.
2. We can set some style properties on a dataframe to see the color coding and understand it better.

```text
df = pd.DataFrame(image[i:j,i:j])
df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')
```

1. np.array\(img\) → Numpy.Array: Numpy’s array function will take an image and return the pixels of the image in a two dimensional data structure called Array.
2. tensor\(img\) → PyTorch.Tensor: PyTorch’s tensor function will take an image and return the pixels of the image in a multi-dimensional data structure called Tensor.
3. List Comprehensions in python returns each value in the given list based on a optional condition when passed to any function f\(\).

```text
new_list = [f(o) for o in a_list if o>0]
 - a_list: The list we want to perfrom the opertaions.
 - o > 0: A optional condition which every element will obey in this 
          case greater than zero.
 - f(o): Something to do with each element

Example:
seven_tensors = [tensor(Image.open(o)) for o in sevens]
sevens: Our list of images of sevens
tensor(Image.open(image)): A function that needs to be executed on each image, 
                           in this case opening the image & creating it as a
                           tensor.

```

1. show\_image\(tensor\) → Image: A fastai function which takes an image tensor & display the image.
2. torch.stack\(list\_of\_tensors\) → PyTorch.Tensor: A Pytorch function that takes a list of tensors \(2-dim image tensors\) and create a 3-dimensional single tensor \(rank-3 tensor\). This will be useful to compute the average across all the images.

![Screen Shot 2021-07-06 at 7 18 21 PM](https://user-images.githubusercontent.com/14807933/124732996-ff234f00-ded0-11eb-8d2b-b0e53717e6e0.png)

10.torch.tensor.float\(\) → float\_value: Casting the values from int to float in PyTorch gives the ability to calculate the mean.

1. L1 norm can be performed using the following

```text
dist_3_abs = (any_image_of_3 - ideal_3).abs().mean()

Note: ideal_3 is the 3 calculated by compuitng mean of the stacked rank-3 
tensor.
```

1. L2 norm can be performed using the following

```text
dist_3_sqr = ((any_image_of_3 - ideal_3)**2).mean().sqrt()
```

1. The above distances can be also computed using the inbuilt PyTorch lib functions from torch.nn.functional package which is by default imported as F by fastai as recommended.

```text
L1 norm:
F.l1_loss(any_3.float(),mean7)

L2 norm:
F.mse_loss(any_3,mean7).sqrt()
```

**Some miscellaneous Key-points:**

1. All Machine learning datasets follow a common layout having separate folders for training and validation \(test\) set’s.
2. A Numpy array & a PyTorch tensor are both multi-dimensional arrays and have similar capabilities except that the Numpy doesn’t have GPU support where as PyTorch does.
3. PyTorch can automatically calculate derivates where as Numpy will not which is a very useful feature in terms of deeplearning.

