# GPT2 Training from Scratch

This repository contains code for training a GPT2 language model from scratch using internet data. The implementation is based on the PyTorch framework and utilizes the Hugging Face Transformers library.

## Overview

The code implements a GPT2 model architecture, including the following key components:

- `GPT2Attention`: Implements the multi-head self-attention mechanism.
- `GPT2MLP`: Implements the feed-forward neural network used in each transformer block.
- `GPT2Block`: Combines attention and MLP layers to form a single transformer block.
- `GPT2Model`: The main model architecture, combining multiple GPT2Blocks.
- `GPT2LMHeadModel`: Adds a language modeling head on top of the GPT2Model for next token prediction.

## Requirements

- Python 3.7+
- PyTorch 1.7+
- Transformers library from Hugging Face

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/matt7salomon/GPT2_from_scratch.git
   cd GPT2_from_scratch
   ```

2. Install the required packages:
   ```
   pip install torch transformers
   ```

## Usage

To train the model, you'll need to:

1. Prepare your internet data and create a custom dataset.
2. Set up the training configuration.
3. Instantiate the `GPT2LMHeadModel`.
4. Define your training loop.

Here's a basic example of how to use the model:

```python
from gpt import GPT2LMHeadModel
from transformers import GPT2Config

# Define the model configuration
config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12
)

# Instantiate the model
model = GPT2LMHeadModel(config)

# Your training loop here
# ...
```

## Customization

You can customize various aspects of the model by modifying the `GPT2Config` parameters, such as:

- `vocab_size`: Size of the vocabulary
- `n_positions`: Maximum sequence length
- `n_embd`: Dimensionality of the embeddings and hidden states
- `n_layer`: Number of transformer blocks
- `n_head`: Number of attention heads

## How it works

We basically start from an empty file and work our way to a reproduction of the GPT-2 (124M) model. If you have more patience or money, the code can also reproduce the GPT-3 models. While the GPT-2 (124M) model probably trained for quite some time back in the day (2019, ~5 years ago), today, reproducing it is a matter of ~1hr and ~$10. You'll need a cloud GPU box if you don't have enough, for that I recommend Lambda.

Note that GPT-2 and GPT-3 and both simple language models, trained on internet documents, and all they do is "dream" internet documents. So this repo/video this does not cover Chat finetuning, and you can't talk to it like you can talk to ChatGPT. The finetuning process (while quite simple conceptually - SFT is just about swapping out the dataset and continuing the training) comes after this part and will be covered at a later time. For now this is the kind of stuff that the 124M model says if you prompt it with "Hello, I'm a language model," after 10B tokens of training:
```
Hello, I'm a language model, and my goal is to make English as easy and fun as possible for everyone, and to find out the different grammar rules
Hello, I'm a language model, so the next time I go, I'll just say, I like this stuff.
Hello, I'm a language model, and the question is, what should I do if I want to be a teacher?
Hello, I'm a language model, and I'm an English person. In languages, "speak" is really speaking. Because for most people, there's
```

## Contributing

Contributions to improve the code or extend its functionality are welcome. Please feel free to submit pull requests or open issues for any bugs or feature requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This implementation is inspired by the GPT2 architecture described in the paper "Language Models are Unsupervised Multitask Learners" by Alec Radford et al. and the Hugging Face Transformers library.

## Disclaimer

Please ensure you have the necessary rights and permissions to use any internet data for training. Be aware of and comply with all relevant laws and regulations regarding data usage and model training.




