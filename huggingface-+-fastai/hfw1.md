# Hugging Face + FastAI - Session 1

**Published:** July 17, 2021

**FastAI + HF Learnings - Week -1**

Setting up a work Environment:

* For using colab, Installation is pretty straight forward and can be done using the pip installer like below: \`\`\` !pip install transformers\[sentencepiece\]
  * \[sentencepiece\] will make sure that all the necessary libraries are included. \`\`\`
* For using local installations use conda / mamba and then utilize the fastconda channel to grab all the necessary fastai related libraries. \(All the necessary links at the end of the blog post\).

Source of this blogpost:

* The great course \(part 1/3 - Introduction\) released & offered by the Hugging face is the source of this blogpost . ![Screen Shot 2021-07-18 at 5 36 58 AM](https://user-images.githubusercontent.com/14807933/126073225-27db212a-cd43-4e36-a337-7bbb2f352099.png)
* This blogpost is my understanding of the Transformers library after participating and learning from the _HuggingFace Course with FastAI bent_ - session 1, generously organized by Weights & Biases and weight lifted by some great folks like Wayde Gilliam, Sanyam Bhutani, Zach Muller, Andrea & Morgan. Sorry if I have missed anyone but thank you all for great hardworking for brining this to the masses.

Keypoints:

* “_Natural Language processing_ is the field of linguists and machine learning focused on understanding everything related to human language” and some of the common tasks include Sequence classification, Token classification, Text generation and Extractive Q & A.
* The state of the art techniques for above tasks comes from the deep learning and transformer models are part of it.

Insights into 🤗 Transformers library:

* “The [🤗 Transformers library](https://github.com/huggingface/transformers) provides the functionality to create and use those shared models.” “The [Model Hub](https://huggingface.co/models) contains thousands of pretrained models that anyone can download and used.”
* The _pipeline_ function is the highest level of the Transformers API and it returns an end-to-end object that can perform a specific NLP task on one or several texts.
* The pipeline is the most basic object in the 🤗 Transformers library and can connect a model with its necessary preprocessing and post-processing steps, allowing us to directly input any text and get an intelligible answer.
* Some of the key steps a pipeline can perform:
  * Preprocess the text into a format the model can understand.
  * Pass the preprocessed inputs to the model.
  * Post-process the predictions of the model so they are humanly understandable.
* For tokenization purpose, the transformers library always encourages using AutoModels which under the hood knows which exact models to utilize for the current task.
* Example of the tokenization:

  Step 1: Create the Tokenizer and choose the architecture ![Screen Shot 2021-07-18 at 6 04 32 AM](https://user-images.githubusercontent.com/14807933/126073340-9ba7ae93-38d1-47a1-9d98-e4592a1dc282.png)

  * The AutoTokenizer knows which exact tokenizer object to create. In this case it has created the BertTokenizerFast Object.
  * The transformers library when creating these objects, has some cool stuff under the hood like vocab\_size which denotes the total size of that model vocabulary set, the max length of that model, which special tokens are included and whether we have any masks included.
  * A Mask is there some of the key capabilities of the transformers library are included.

Step 2: Inputing a text to the tokenizer ![Screen Shot 2021-07-18 at 6 15 41 AM](https://user-images.githubusercontent.com/14807933/126073293-18b37feb-f949-413f-8d77-0bb04e35dfd8.png)

* The tokenizer model will create the numerical ids of the sentence we have provided and also the token type ids which are essential for the Bert model.
* The token type ids are essentially a way for the Bert models to identify two sentences.
* The attention masks are a special capability of the transformers library by which it determines whether to apply the learnings / representation to the other tokens from the current tokens it is looking at.

Additional Step: \* We can decode these token to see the exact representation of how the words are created as tokens. ![Screen Shot 2021-07-18 at 6 27 59 AM](https://user-images.githubusercontent.com/14807933/126073353-d4972697-b3f8-4331-b029-9b90be29405e.png)

* The CLS & SEP are special tokens added by the model and vary by the model we are using. And also some special tokenization like \#\# depending on how they utilize the sub-word tokenization.
* A sub-word tokenization is a strategy by which the model limits the size of the learning batches for performance.

Example of using a Pipeline:

The beauty of the pipeline API lies in the simplicity of steps we have to apply a model and get a result.

Step1: Create a classifier

* Create a classifier according to the task. For instance a sentiment-analysis classifier and provide a text to classify its sentiment. ![Screen Shot 2021-07-18 at 6 41 34 AM](https://user-images.githubusercontent.com/14807933/126073358-7da6761b-86f9-49c4-aef3-8f6efbd18138.png)

Step2: Utilizing the classifier object ![Screen Shot 2021-07-18 at 6 42 36 AM](https://user-images.githubusercontent.com/14807933/126073366-11bad535-6108-4b5a-b0c3-d9dc8958c43b.png)

Additional Step 1: Know its information. ![Screen Shot 2021-07-18 at 6 43 29 AM](https://user-images.githubusercontent.com/14807933/126073372-f621ab2e-a482-46bf-bd82-087d606c17a0.png)

* The distil models of the bert are the smaller versions of the architecture for faster performance. Additional Step 2: Using the user-contributed models
* For using the user-contributed models, we need to specify the model while creating the classifier object in step 1 ![Screen Shot 2021-07-18 at 6 48 20 AM](https://user-images.githubusercontent.com/14807933/126073378-f17a05db-37ee-475b-a0ba-d768c99bebc6.png)

Additional Step 3: Using Zero-shot classification

* A zero-shot classification refers to a specific use case of machine learning \(and therefore deep learning\) where you want the model to classify data based on very few or even no labeled example, which means classifying on the fly.
* The below picture shows the difference between the transfer learning technique that we utilize & zero-shot learning.

![Screen Shot 2021-07-18 at 6 53 23 AM](https://user-images.githubusercontent.com/14807933/126073385-0cd74048-b839-4f77-9091-df5054b395d2.png)

* The below code snippet is a classic example of how to start with zero-shot classification ![Screen Shot 2021-07-18 at 6 54 58 AM](https://user-images.githubusercontent.com/14807933/126073390-e4f0fc98-d7c8-471e-93cd-da9bf435ef67.png)

Example of Text-Generation capabilities of Pipeline API:

* Create the pipeline object with the text-generation task and provide the piece of text that you want the generator to complete the rest of the text for you. ![Screen Shot 2021-07-18 at 6 59 56 AM](https://user-images.githubusercontent.com/14807933/126073406-51335886-f85c-4150-ac82-6a360e3f86a0.png)
* We also have the capability to control the text generation via parameters like max\_length which restricts the length of the sentence and num\_return\_sequences which will return the exact number of sequences. 

Example of a Language Modeling:

* Language modeling \(LM\) is the use of various statistical and probabilistic techniques to determine the probability of a given sequence of words occurring in a sentence. ![Screen Shot 2021-07-18 at 7 09 26 AM](https://user-images.githubusercontent.com/14807933/126073439-0d42d8f8-f466-4511-993b-27698e738bb0.png)
* We can utilize the “fill-mask” architecture from the library to achieve such amazing NLP capability. ![Screen Shot 2021-07-18 at 7 11 17 AM](https://user-images.githubusercontent.com/14807933/126073451-c94dec33-ff73-40e8-a220-6c5cab2869fb.png)

Example of Token classification:

* We have several use cases in token classification and one of such amazing capability is Named-entity recognition \(NER\).
* One the capabilities of the NER is that, it can group tokens after tokenization to understand which ones are together and if they are names are not etc. ![Screen Shot 2021-07-18 at 7 14 28 AM](https://user-images.githubusercontent.com/14807933/126073459-42dadc23-bc23-4659-9885-e47025d11de1.png)

Example of Question-Answering Task:

* The transformer library is pretty good at Question-Answering tasks like extracting the answer from the provided question and its context.
* Here we are creating a pipeline providing the task “question-answering” and then provide the question and context to the pipeline object which will then extract the answer from the context and return it. ![Screen Shot 2021-07-18 at 7 20 05 AM](https://user-images.githubusercontent.com/14807933/126073468-d94aa71f-773b-46c9-a393-b21e3b26c75a.png)

Example of Summarization capabilities of the library:

* We can summarize the text by using the summarization capabilities of the Pipeline API by providing the “summarization” task. ![Screen Shot 2021-07-18 at 7 26 49 AM](https://user-images.githubusercontent.com/14807933/126073474-66d3356b-6964-49a7-b231-9cbc2b54c769.png)
* And the summarizer object will then return the summary of the text provided like below: ![Screen Shot 2021-07-18 at 8 58 44 AM](https://user-images.githubusercontent.com/14807933/126074013-29fefbbe-3f71-4186-89f8-707d589e6dd5.png)

Example of Translation capabilities of the API:

* We can utilize the translation task with the pipeline object to create a translator which will translate the provided text from the language given to language needed by looking at the model attribute. ![Screen Shot 2021-07-18 at 7 30 43 AM](https://user-images.githubusercontent.com/14807933/126073490-c44b3bcb-83d0-4e78-a586-6f3afd02af41.png)

Pretty Cool 🤗.

We have looked at the Transformers library and its extraordinary capabilities for different types of NLP tasks. Now this my understanding about what a transformer is and how does that actually work.

* _A transformer_ is a deep learning model that adopts the mechanism of attention, differentially weighing the significance of each part of the input data. It is used primarily in the field of natural language processing and in computer vision.
* In the case of HF🤗 all the Transformer models mentioned above \(GPT, BERT, BART, T5, etc.\) have been trained as _language models_.
* If a model excels at the language modeling task then, we can use the transfer learning technique where these pre-trained language models can be fine tuned for specific task.
* The transformer architecture is originally built to handle translation.
* In simple summary, the basic transformer under the hood has two steps, encode and decode. The encoder is focused on understanding the input and the decoder is focused on generating the output or a higher level representation of the output which we can utilize for predictions. ![Screen Shot 2021-07-18 at 7 54 34 AM](https://user-images.githubusercontent.com/14807933/126073494-20d9f39a-9eb6-4c6a-b94e-799322ac7c35.png)
* One of the key components of the transformers working is “Attention”.
* Attention allows the model to focus on the relevant parts of the input sequence as needed.
* When the model is processing the text \(words\), self attention allows it to look at other positions in the input sequence for clues that can help lead to a better encoding for this word.
* Some of the transformer models are encoder only like ALBERT, BERT, DistilBERT, ELECTRA and RoBERTa.
* Some of the transformer models are decoder only and focus on text generation like CTRL, GPT, GPT-2 & Transformer XL.
* Some of the transformer models are Sequence-to-Sequence and utilize both encoder & decoder. These models work well for tasks where input distribution is different from output distribution in tasks such as summarization, translation, generative Q&A . Some the examples are BART / MBART, M2M100, MarianMT, Pegasus, PropheNet, T5/mT5.
* One of the key-consideration to take into account while using these models is the bias. So we should be aware of that and try to reduce it as mush as possible.

