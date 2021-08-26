# Deep Learning for Coders / Chapter-2 / Week-3

**Published:** June 23, 2021

This following notes is from the Week-3 of the FastAI/fastBook reading session hosted by Aman Arora \(Weights & Baises\)

* Important links
  * Note: The slack has been completely moved out of equation due to its limitation of 10k messages in the new version. Now all the
  * [forums.fast](http://forums.fast/).ai - check out the forums and explore the existing topics or ask anything that you are stuck with.
  * [wandb.me](http://wandb.me/)/fc - Weights & Biases forums where we have weekly information on the **Forum tab.**
  * [FastBook-Week-3](https://wandb.ai/aarora/discussions/Fastbook-Reading-Group-Week-3--Vmlldzo3OTMwODk?galleryTag=forum) - This would be the link for the Week 3 on the wandb forums.
  * All my learnings from the session are also posted in my blog and in more detailed fashion at [https://ravichandraveeramachaneni.github.io/](https://ravichandraveeramachaneni.github.io/)

KeyPoints - Chapter 2 \(From Model to Production\)

* For deploying models into production we need : data, a trained model, API’s around the model, nice UI/UX experience \(for services from the browser\), good infrastructure, best coding practices etc.
* There are 4 main categories in a deep learning project before production.
  1. Data Preparation
  2. Labelling data
  3. Model Training
  4. Production
* The better way would be to allocate equal time for each task.
* Underestimating the constraints and overestimating the capabilities of deep learning may lead to frustratingly poor results. So be keen on understanding what is needed.
* Conversely, overestimating the constraints and underestimating the capabilities of deep learning may mean you do not attempt a solvable problem because you talk yourself out of it. So don’t stop yourself from trying a model. Iterate your learnings.
* It’s better to iterate the project end-to-end rather than just fine-tuning the model or making some fancy GUI.
* It’s only by practicing \(and failing\) a lot that you will get an intuition of how to train a model.
* Start learning with the existing examples and the existing domains where deep learning is already applied and then look for more branches.
* There are many accurate models that are of no use to anyone, and many inaccurate models that are highly useful.
* A Drivetrain approach of how to use data not just to generate data but to produce actionable results is shown in the below picture:

![Screen Shot 2021-06-30 at 4 53 10 PM](https://user-images.githubusercontent.com/14807933/124048002-9b7dbb00-d9d2-11eb-83e9-c8e7502f822e.png)

* Below is the cool little gist which shows right from how to make our datasets to training & inference. https://gist.github.com/RaviChandraVeeramachaneni/12b2ed5ef7342048f92a86b019d4fd2f
* Some of the problems to understand while building data centric products with deep learning involved:

  * Understanding and testing the behavior of a deep learning model is much more difficult than with most other code we write.
  * The neural network’s behavior emerges from the model’s attempt to match the training data, rather than being exactly defined. So this could be a disaster.
  * Out-of-domain data and domain shifts are another problem to be considered.
  * One possible approach outlined to understand the problems would be best described by below Image.

  ![Screen Shot 2021-06-30 at 5 14 34 PM](https://user-images.githubusercontent.com/14807933/124048029-a9334080-d9d2-11eb-98b7-010ec8f44dc8.png)

