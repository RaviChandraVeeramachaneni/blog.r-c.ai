# Hugging Face + FastAI

 **Published:** August 02, 2021

**Training Transformers with FastAI**

**Contents:**

1. Source of this blogpost
2. Setting up a work Environment
   1. Environment
   2. Necessary Libraries
   3. Necessary imports
3. Do we need FastAI / Is 🤗 libraries not sufficient
   1. Highlights of 🤗
   2. Highlights of FastAI
4. Blurr API Highlights
5. Credits

**Source of this blogpost:**

* This blogpost is my understanding of the Transformers library after participating and learning from the _HuggingFace Course with FastAI bent_ - generously organized by Weights & Biases and weight lifted by some great folks Wayde Gilliam, Sanyam Bhutani, Zach Muller, Andrea Pessl & Morgan McGuire. Sorry, if I have missed anyone, but thank you all for the great hard work to bring this to the masses.
* Also this blogpost is submitted as part of the blogpost competition held by the team which was announced in session-3.

**Setting up a work Environment:**

**Environment:**

* For using local installations use conda/mamba and then utilize the fastconda channel to grab all the necessary fastai related libraries.
* While learning this course, I am using google colab pro and this needs few settings before we get started.
* First thing we need is to set the Runtime-Type, so for this setting, navigate to the colab -&gt; Runtime Tab and click on “Change Runtime Type”.

![Screen Shot 2021-08-02 at 2 14 24 PM](https://user-images.githubusercontent.com/14807933/127965981-045534c7-3c26-4c17-acb5-20901b8e1364.png)

* On the “Change Runtime Type” pop-up select the below options.

![Screen Shot 2021-08-02 at 2 16 14 PM](https://user-images.githubusercontent.com/14807933/127966007-e89f095e-7872-4d74-a09c-d4be0ccc87e3.png)

* You are all set as per environment.

**Necessary Libraries:**

* Before we import anything, we need to install the required libraries. The following lines are needed as per our requirement on whether we are using fastai, HFTransformers, blurt api, adaptnlp api or fast hugs etc.

  ```text
  !pip install -U fastai
  !pip install transformers[sentencepiece] 
  !pip install ohmeow-blurr
  !pip install git+https://github.com/novetta/adaptnlp@dev
  !pip install -qq git+git://github.com/aikindergarten/fasthugs.git
  ```

* In the following tutorial I am going to explain blurr api which is a wrapper on top of fastai to use 🤗 transformers.

**Necessary Imports:**

* We need to import the necessary libraries as needed. But always a best practice is to keep all the imports on top of the notebook to keep it clean and readable.

```text
import torch
from fastai.text.all import *

from datasets import load_dataset, concatenate_datasets
from transformers import *

from blurr.utils import *
from blurr.data.core import *
from blurr.modeling.core import *
```

**Do we need FastAI / Is 🤗 libraries not sufficient:**

**Highlights of 🤗:**

* The transformers library is great & the Pipeline API is self sufficient to tokenize, train and infer on the dataset.
* Some of the best things about Pipeline API is the ease of having prebuilt checkpoints, AutoXXX classes like AutoModel, AutoTokenizer etc.
* Example here: [fastai+HF\_week2\_transformers\_example.ipynb · GitHub](https://gist.github.com/RaviChandraVeeramachaneni/f0d9a653417cdec95561e89070aaa2e0)

**Highlights of FastAI:**

* Though we have all the functionality in the 🤗, there are lot of things we can improve and experiment with.
* One of the main advantage of having wrappers like blurr, adapnlp or fast hugs is the flexibility of looking at each step and customize as per requirement.
* We have lots of great tools like learning rate finder, augmenting techniques, debugging data blocks etc.
* But out of the box if we want to wrap/integrate the external objects like 🤗 transformers we need to build the functions from scratch. This problem is solved by API’s like Blurr.

**Blurr API Highlights:**

The following steps will detail the Blurr API. But to make it easy and understandable we will compare each step with fastai:

* Building an application using Blurr API is mostly similar to building one in fastAI but with very easy and convenient functions.
* Couple of steps like downloading datasets, unzipping, wrapping them in to paths etc can be done in native fastAI functions like below:

  ```text
  path = untar_data(URLs.IMDB)
  files = get_text_files(path, folders = ['train', 'test', 'unsup'])
  ```

* But the Blurr API has lots of convenient functions like `BLURR*.*get_hf_objects` to get the hugging face objects, which are vital in constructing out the further steps in building an application like constructing data blocks.
* In FastAI, tokenization can be done using the following code snippet

  ```text
  spacy = WordTokenizer()
  tkn = Tokenizer(spacy)
  ```

* But in case of integrating 🤗 api we prefer using their config and that can be done using their AutoConfig class like below:

```text
config = AutoConfig.from_pretrained(pretrained_model_name)
```

* Once we have the config, we can have our Blurr magic happen with below code snippet:

```text
hf_arch, hf_config, hf_tokenizer, hf_model = BLURR.get_hf_objects(pretrained_model_name, model_cls=model_cls, config=config)
```

* In the above step, the `get_hf_objects` will return the model, config, arch and tokenizer in one step.
* The next step would be to construct a data loader. If we are using FastAI then we can do as below:

```text
get_imdb = partial(get_text_files, folders=['train', 'test', 'unsup'])

dls_lm = DataBlock(
    blocks=TextBlock.from_folder(path, is_lm=True),
    get_items=get_imdb, splitter=RandomSplitter(0.1)
).dataloaders(path, path=path, bs=128, seq_len=80)
```

* But if we want to do that using Blurr API, its lot more easier and convenient like below:

```text
blocks = (HF_TextBlock(hf_arch, hf_config, hf_tokenizer, hf_model), CategoryBlock)
dblock = DataBlock(blocks=blocks,  get_x=ColReader('text'), get_y=ColReader('label'), splitter=ColSplitter())

dls = dblock.dataloaders(imdb_df, bs=4)
```

* So the next step would be to create a Learner object which is our crucial step. In Fast AI & Blurr API we can achieve with below code snippet:

```text
#FastAI
learn = language_model_learner(
    dls_lm, AWD_LSTM, drop_mult=0.3, 
    metrics=[accuracy, Perplexity()]).to_fp16()
learn.fit_one_cycle(1, 2e-2)
```

```text
#Blurr API
learn = Learner(dls, 
                model,
                opt_func=partial(Adam, decouple_wd=True),
                loss_func=CrossEntropyLossFlat(),
                metrics=[accuracy],
                cbs=[HF_BaseModelCallback],
                splitter=hf_splitter)
learn.freeze()
learn.fit_one_cycle(3, lr_max=1e-3)
```

* And the rest of the steps like finding learning rate, freezing, unfreezing, or getting predictions can be done in native fastAI using below:

```text
#Freezing 
learn.freeze()

#UnFreeze
learn.unfreeze()

#validation
learn.validate()

#Predictions, losses
preds, targs, losses = learn.get_preds(with_loss=True)
```

**Credits:**

* The following are the resources I have referred when learning through this entire journey of creating blogpost, learning fastai, hugging face etc.
* fastai: [Practical Deep Learning for Coders](https://course.fast.ai/)
* walkwithfastai: [Walk with fastai](https://walkwithfastai.com/)
* hugging face: [Hugging Face – The AI community building the future](https://huggingface.co/)
* blurrapi: [blurr](https://ohmeow.github.io/blurr//)
* AdaptNLP: [GitHub - Novetta/adaptnlp: An easy to use Natural Language Processing library and framework for predicting, training, fine-tuning, and serving up state-of-the-art NLP models.](https://github.com/Novetta/adaptnlp)
* Fasthugs: [GitHub - morganmcg1/fasthugs: Use fastai-v2 with HuggingFace’s pretrained transformers](https://github.com/morganmcg1/fasthugs)
* Hugging Face + FastAI course: [Weights & Biases](https://wandb.ai/wandb_fc/events/reports/W-B-Study-Group-Lectures-fast-ai-w-Hugging-Face--Vmlldzo4NDUzNDU?galleryTag=events)
* And many many more blogpost, YouTube videos and mainly the colabs shared by lots of great people.

**Link to my other blogposts:**

* Link to the blogposts I have wrote while learning this course: [Hugging Face + FastAI - Session 1 - Ravi Chandra Veeramachaneni](https://ravichandraveeramachaneni.github.io/posts/bp7/), [Hugging Face + FastAI - Session 2 - Ravi Chandra Veeramachaneni](https://ravichandraveeramachaneni.github.io/posts/hfw2/)
* Link to the blogposts I have wrote while learning the FastAI: [Blog posts - Ravi Chandra Veeramachaneni](https://ravichandraveeramachaneni.github.io/posts/)

