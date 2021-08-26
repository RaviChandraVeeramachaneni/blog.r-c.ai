# Deep Learning for Coders / Chapter-4\(Part-2\) / Week-5

**Published:** July 07, 2021

KeyPoints from Chapter-4 2nd Half \(Training a Digit Classifier\)\*

1. **Stochastic Gradient Descent** is the way of automatically updating the weights of the neural network input’s, based on the previous result to gain the maximum performance or in simple terms better accuracy.
2. This can be made entirely automated, so that the network can reach back to the initial inputs, update their weights and can perform the training again with the new weights. This process is called the back-propagation.
3. Example of an function that can be used to classify a number based on the above described way:

   ```text
   def pr_eight(x,w):
    return (x*w).sum()		
   ```

   In the above function, x - vector representation of the input Image. w - vector of weights

**How can we make this function into a Machine Learning Classifier:**

1. Initialize the weights.
2. For each image use these weights to predict whether a 3 or a 7.
3. Calculate the loss for this model based on these predictions.
4. Calculate the gradient, which helps to determine the change in the weight and in turn the loss for that weight. And this has to be done for each weight.
5. Change the weights based on the above gradient calculation. This step is called “Step”.
6. Now we need to repeat from prediction step \(step 2\).
7. Iterate until your model is good enough.

![Screen Shot 2021-07-12 at 4 12 08 PM](https://user-images.githubusercontent.com/14807933/125567801-065e6247-41fa-45bf-8f97-664037f5d26f.png) **Disclaimer: All the Images in this blogpost are from the FastBook**

**Detailing the each step in the above process:**

1. **Initialize**: Initializing the parameters/ Weights to random values will perfectly work.
2. **Loss**: We need a function that return the loss in terms of a number. A good model has small loss and vice versa.
3. **Step**: We need to determine whether to increase the weights or decrease the weights to maximize the performance or in other terms minimize loss. Once we determine the increase or decrease then we can increment/decrement accordingly in small amounts and check at which point we are achieving the maximum performance. This process is manual and slower and can be automated and achieved by calculating gradient using calculus. Gradient calculation will figure out directly whether to increment / decrement weights and by how much amount.
4. **Stop**: This is where we will decide & implement about number of epochs to train our model . In the case of digit classifier, we will train our model until over fitting \(Our model performance gets worse\) occurs.

Example of a simple loss function and understand about slope

```text
def f(x):
'''
	Simple quadratic loss function
	x: weight parameter
'''	
	return x**2
```

1. Visualizing the above function with _slope_ at one point , when initialized it with a random weight parameter.

![Screen Shot 2021-07-12 at 5 28 23 PM](https://user-images.githubusercontent.com/14807933/125568002-99d55f41-e06e-47d1-9c50-c7f1791e0327.png)

1. Once we determine the direction of the slope, then we can keep adjusting the weight, calculate the loss every time and repeat the process until we reach the lowest point on the curve where the loss is minimum.

![Screen Shot 2021-07-13 at 1 16 48 PM](https://user-images.githubusercontent.com/14807933/125568059-e3a3067c-c3e3-469f-a478-184c721d93cc.png)

1. For gradient calculation, reason behind using calculus over doing it manually is to achieve performance optimization.

**Calculating Gradients, Derivatives & why do we need them:** **What & Why:**

1. In simple words, gradient will tell us how much each weight has to be changed to make our model better. And it is a vector and it points in the direction of steepest ascent to minimize the loss.
2. A derivate of a function is a number which tells us, how much a change in parameter will change the result of the function.
3. For any quadratic function, we can calculate its derivative.
4. Important Note: A derivative is also a function, which calculates the change, rather than a value like a normal function does.
5. So, we need to calculate gradient to know how the function will change with a given value so that we can try to reduce the function to the smallest number where the loss is minimum.
6. And the computational shortcut provided by calculus to do the gradient calculation is called Derivative.

**How to we calculate derivates:**

1. We need to calculate gradient for every weight since we don’t have just one weight.
2. We will calculate the derivative of one weight considering the others as constants and then repeat the process for every other weight.
3. A pytorch example to calculate derivative at value 3

   ```text
   xt = tensor(3.).requires_grad_()		
   ```

4. In deep learning, the “gradients” usually means the value of a function’s derivative at a particular argument value.

**Example of a gradient calculation:** Consider we want to calculate derivative of x_\*2 and the result is 2_x, where x=3, so the gradient must be 2\*3 = 6 ![Screen Shot 2021-07-13 at 2 34 22 PM](https://user-images.githubusercontent.com/14807933/125568513-659de6ca-35ea-44c5-9e48-716f2635b8a9.png)

1. The gradient only tell us the slope of the functions and not exactly how much weight we have to adjust. But the intuition is if we have a big slope then we need to make lot of adjustments to weights and vice versa if the slope is small then we are almost close to optimal value.

**How do we change Weights/Parameters based on Gradient Value:**

1. The most important part of the deep learning process is to decide how to change the parameters based on the gradient value.
2. Simplest approach is to multiply the gradient with a small number often between 0.001 and 0.1 \(but not limited to this range\) and this is called **Learning Rate\(LR\)**.
3. Once we have a Learning Rate we can adjust our parameters using the below function
4. This process is called as stepping the parameters using the Optimizer step. This is because, in this step we are trying to find an optimal weight.
5. We can pick either very low learning rate or a very high learning rate and both have their consequences.
6. If we have a low learning rate then we have to do lot of steps to get the optimal weight.

![Screen Shot 2021-07-13 at 3 42 13 PM](https://user-images.githubusercontent.com/14807933/125568625-82d3eb7a-056f-479a-b747-3129e315012d.png)

1. Picking a very high learning rate is even worse and can result in the loss getting worse\(left image\) or may bounce around\(right image, requiring lot of steps to settle down\) . So we are loosing our goal of minimizing the loss.

![Screen Shot 2021-07-13 at 3 47 29 PM](https://user-images.githubusercontent.com/14807933/125568681-05c9d7d6-1e18-4d7d-9a06-fdd59ff919b1.png)

**Conclusion**: We are trying to find the minimum \(loss\) using the SGD and this minimum can be used to train a model to fit the data better for our task.

