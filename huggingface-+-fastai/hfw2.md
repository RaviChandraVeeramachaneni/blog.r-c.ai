# Hugging Face + FastAI - Session 2

**Published:** July 24, 2021

**FastAI + HF Learnings - Week -2**

**Contents:**

1. Setting up a work Environment
2. Source of this blogpost
3. Keypoints
   1. Behind the pipeline
      1. Tokenizing
      2. Model
      3. Post-Processing
   2. “Transformers” Tokenizers
      1. Word-based tokenizers.
      2. Character-based tokenizers.
      3. Subword tokenizers.
   3. “Transformers” Models
   4. Handling Multiple Sequences
      1. Padding
      2. Truncation
      3. Attention Mask.
4. Links for the Github Gists
5. Credits

**Setting up a work Environment:**

* For using colab, Installation is pretty straight forward and can be done using the pip installer like below: \`\`\` !pip install transformers\[sentencepiece\]
  * \[sentencepiece\] will make sure that all the necessary libraries are included. \`\`\`
* For using local installations use conda / mamba and then utilize the fastconda channel to grab all the necessary fastai related libraries. \(All the necessary links at the end of the blog post\).

**Source of this blogpost:**

* The great course \(part 1/3 - Introduction\) released & offered by the Hugging face is the source of this blogpost .

![Transformer\_intro](https://user-images.githubusercontent.com/14807933/126880601-ed809a53-d6f5-4ed0-8ae3-36a4556dfca3.png)

* This blogpost is my understanding of the Transformers library after participating and learning from the _HuggingFace Course with FastAI bent_ - session 2, generously organized by Weights & Biases and weight lifted by some great folks like Wayde Gilliam, Sanyam Bhutani, Zach Muller, Andrea & Morgan. Sorry if I have missed anyone, but thank you all for the great hard work to bring this to the masses.

**Keypoints:**

**Behind the Pipeline:** \* At a high-level in the Pipeline API, we have 3 things going on: 1. Tokenizing 2. Model 3. Post-processing.

![Transformer\_steps](https://user-images.githubusercontent.com/14807933/126880611-4611d416-75be-4172-aa2c-41d534a04266.png)

**Step 1: Tokenizing**

* Tokenizing the raw text \( We do this step because the model needs the numerical representation of the text\).
* The Tokenizer splits the words into subwords called tokens.
* We have different tokenizers in the HF library.
* Each of these tokens are mapped to an Integer and these integers are run through the model as a list or dictionary.
* We need to build a tokenizer using the same checkpoint used for the model.

> Checkpoints capture the exact value of all parameters used by a model. Checkpoints do not contain any description of the computation defined by the model and thus are typically only useful, when source code uses available saved parameter values.

![tokens](https://user-images.githubusercontent.com/14807933/126880648-852326a8-7bed-4c99-9f39-78033eccf618.png)

* There are multiple rules that can govern the tokenization process, which is why we need to instantiate the tokenizer using the same checkpoint of the model, to make sure we use the same rules that were used when the model was pre-trained.
* In simple words if we want to take advantage of the pre-trained checkpoint\(model\), it’s better to use the same tokenizer they have used for pre-training the model.
* Example of tokenizing a sequence. We can check the tokens and then we can decode the tokens to cross check those tokens.

```text
Step 1: Raw text -> Tokens
sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)
# ['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']

Step 2: Tokens -> Input IDs
ids = tokenizer.convert_tokens_to_ids(tokens)
# [7993, 170, 11303, 1200, 2443, 1110, 3014]

Step 3: Input IDs -> Raw text
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
# 'Using a Transformer network is simple'
```

**Step2: Model**

* The input to the model are typically the tokens \(numerical representations of the raw text\) and some times special things like attention masks, token type id’s based on the auto-tokenizer we are trying to use.

![Model](https://user-images.githubusercontent.com/14807933/126880664-f870410a-d18f-4948-90c0-59d1197248ea.png)

> ```text
> Note: Please read about AutoTokenizer example in this earlier week-1 blogpost [Hugging Face + FastAI - Session 1 - Ravi Chandra Veeramachaneni](https://ravichandraveeramachaneni.github.io/posts/bp7/) * To create a model with a pertained checkpoint we can use the below code snippet ``` model = AutoModel.from_pretrained(checkpoint)	 ``` * The Automodel’s output will be a feature vector for each of those tokens called “hidden states / features” The Features are simply what model has learned from that input tokens. * These features are now passed to the specific part of the model called “head” and this head would be different based on tasks like Summarization, Text-generation etc. 
> ```

**Step3:PostProcessing**

* The Outputs from the model are similar to the named tuples or dictionaries called logits \(predictions\).
* In the post-processing step, these logits will be fed to activation function like SoftMax with loss function like cross -entropy. And we will have our final output.

> Softmax function takes the predictions from model as input and outputs the same as probabilities between 0 and 1 and they all sum upto one.

> Cross-Entropy loss is use of negative loss on probabilities. \(Or in simple terms\) Cross-Entropy loss is a combination of using the negative log likelihood on the log values of the probabilities from the softmax function.
>
> Read more about SoftMax function, cross-entropy loss in detail here in this blogpost: [Deep Learning for Coders / Chapter-5 / Week-6 - Ravi Chandra Veeramachaneni](https://ravichandraveeramachaneni.github.io/posts/bp8/)

![Cross-Entropy\_loss](https://user-images.githubusercontent.com/14807933/126880677-8a592899-908d-4781-b307-fb5371a63306.png)

> Please check this github gist where I tried learning all the steps above as show cased in the session: [fastai+HF\_week2\_transformers\_example.ipynb · GitHub](https://gist.github.com/RaviChandraVeeramachaneni/f0d9a653417cdec95561e89070aaa2e0)

**“Transformers” Tokenizers:**

* A powerful and simple class from 🤗 for creating a tokenizer is `AutoTokenizer` and it utilizes the `from_pretrained` method to do that.

  ```text
  checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained(checkpoint)	
  ```

> Please check this github gist where I have tried creating a Tokenizer from scratch: [fastai+HF\_week2\_Tokenizer\_from\_scratch.ipynb · GitHub](https://gist.github.com/RaviChandraVeeramachaneni/fd5b2c1397626f3b44074f5a53008b56)

* There are different types of tokenizers:
  1. Word-based tokenizers.
  2. Character-based tokenizers.
  3. Subword tokenizers.

**Word-based Tokenizer:**

* A word-based tokenizer is a very basic tokenizer that splits on spaces and punctuation.

![word\_based](https://user-images.githubusercontent.com/14807933/126880684-67fbcd81-f7e1-4f8a-a742-2d9a35d22f1e.png)

* These tokenizer’s are easy to setup with few rules and often yields decent results.
* However the drawbacks of these tokenizers are that they create large vocabularies with a possibility of unknown tokens which would cause loss of meaning.

**Character-based Tokenizer:**

* A character-based tokenizer splits based on characters as the name suggest.

![character](https://user-images.githubusercontent.com/14807933/126880690-d89b80ed-9c38-446f-b493-305b3ce4dfdf.png)

* These tokenizer’s have a very small vocabulary and very few OOV \(out of vocabulary\) words or unknown words.
* However the drawbacks of these tokenizers are that, they have very less meaningful embeddings and hence requires more tokens to represent a sequence.

**Subword Tokenizers:**

* Subword tokenization algorithms rely on the principle that frequently used words should not be split into smaller subwords, but rare words should be decomposed into meaningful subwords.

![subword](https://user-images.githubusercontent.com/14807933/126880695-b05c868d-7f3b-4ac3-8c83-4ea70777e080.png)

* The way of splitting the words provides a lot of semantic meaning, requires smaller vocabulary with very very less unknown words and works well with different languages.

**“Transformers” Models:**

* A powerful and simple class from 🤗 for creating a model is `AutoModel` which “can automatically guess the appropriate model architecture for your checkpoint, and then instantiates a model with this architecture.”
* The `from_pretrained`method that is used to take the checkpoint and output the model.

  ```text
  from transformers import AutoModel
  model = AutoModel.from_pretrained(checkpoint)
  ```

* After this step, the model is now initialized with all the weights of the checkpoint. It can be used directly for inference on the tasks it was trained on, and it can also be fine-tuned on a new task.
* We can also save this trained model to our local machines using `save_pretrained` method. This method will output two files a config.json which has metadata of transformers version, checkpoint information etc. and a pytorch\_model.bin file which contains our model weights.

**Handling Multiple Sequences:**

When we are handling multiple sequences, the following things need to be considered:

1. Padding
2. Truncation
3. Attention Mask

**Padding:**

* When we are sending inputs to a model we send them in mini-batches and each batch would be a square matrix even though the length of tokens vary from batch to batch.
* We need to pay attention to the padding tokens and ignore them.
* We add padding tokens to every batch to make sure they are of same length and there are different ways to add these padding tokens like
  1. Longest
  2. max\_length
  3. max\_length with length specified.

```text
# Will pad the sequences up to the maximum sequence length in the batch
model_inputs = tokenizer(sequences, padding="True|longest")

# Will pad the sequences up to the model max length (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")

# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
```

**Truncation:**

* Truncation will make sure that the batches we send to the model will fit the architecture or by the GPU.
* We can handle the truncation by specifying the attribute `truncate` and also we can specify the `max_length` to limit the sequence length.

```text
# Will truncate the sequences that are longer than the model max length (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, truncation=True)

# Will truncate the sequences that are longer than the specified max length
model_inputs = tokenizer(sequences, max_length=8, truncation=True)	
```

**Attention Mask:**

* Attention Mask is a way to specify the model to consider the truncation, padding, max\_lengths etc.

> Attention layers of the transformers model contextualize each token.

* Attention masks are tensors with the exact same shape as the input IDs tensor, filled with 0s and 1s. The 1s indicate that the tokens should be attended and 0s should be ignored.

**Links for the Github Gists:**

1. Behind the pipeline: [fastai+HF\_week2\_transformers\_example.ipynb · GitHub](https://gist.github.com/RaviChandraVeeramachaneni/f0d9a653417cdec95561e89070aaa2e0)
2. Tokenizer from scratch: [fastai+HF\_week2\_Tokenizer\_from\_scratch.ipynb · GitHub](https://gist.github.com/RaviChandraVeeramachaneni/fd5b2c1397626f3b44074f5a53008b56)
3. Previous week \(week-1\) blogpost here: [Hugging Face + FastAI - Session 1 - Ravi Chandra Veeramachaneni](https://ravichandraveeramachaneni.github.io/posts/bp7/)
4. FastBook’s blogpost on Softmax & CrossEntropy: [Deep Learning for Coders / Chapter-5 / Week-6 - Ravi Chandra Veeramachaneni](https://ravichandraveeramachaneni.github.io/posts/bp8/)

**Credits:**

* All images are from the session-2 of the reading group and cross-entropy image is from google images.

