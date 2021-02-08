from transformers import GPT2LMHeadModel, GPT2Model
import torch

class GPT2LMHeadLogit(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.vocab_size

    def __call__(self, x):
        outputs = super().__call__(x)
        logits = outputs[0] #[batch_size, seqlen, vocab_size]
        return logits


class GPT2Featurizer(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.n_embd

    def __call__(self, x):
        outputs = super().__call__(x)
        hidden_states = outputs[0] #[batch_size, seqlen, n_embd]
        return hidden_states


class GPT2FeaturizerLMHeadLogit(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.vocab_size
        self.transformer = GPT2Featurizer(config)

    def __call__(self, x):
        hidden_states = self.transformer(x) #[batch_size, seqlen, n_embd]
        logits = self.lm_head(hidden_states) #[batch_size, seqlen, vocab_size]
        return logits
