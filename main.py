### main.py
from src.train import train_model

if __name__ == "__main__":
    results = train_model(epochs=5, batch_size=32)
    print(results)

### requirements.txt
jax
flax
optax
datasets
transformers
numpy

## 🧠 Model: MiniGPT
A small transformer model with self-attention blocks for sequence modeling.

## 📦 Dependencies

bash
pip install -r requirements.txt


## 🚀 Run Training
---
  bash
python main.py
---

## 📊 Results
After training, the script will output the final average loss and a generated text sample.

---

### src/data.py
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

def load_and_preprocess_data(batch_size=32, max_length=128):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        texts = [t for t in examples["text"] if t and t.strip()]
        if not texts:
            return {}

        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
            return_attention_mask=True
        )

        return {
            "input_ids": np.array(tokenized["input_ids"], dtype=np.int32),
            "labels": np.array(tokenized["input_ids"], dtype=np.int32),
        }

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.filter(lambda x: "input_ids" in x and len(x["input_ids"]) > 0)

    return tokenized_dataset.with_format("numpy"), tokenizer


### src/model.py
import flax.linen as nn

class TransformerBlock(nn.Module):
    num_heads: int

    def setup(self):
        self.attention = nn.SelfAttention(num_heads=self.num_heads)
        self.dense = nn.Dense(256)

    def __call__(self, x):
        attn_output = self.attention(x)
        return self.dense(attn_output)

class MiniGPT(nn.Module):
    vocab_size: int
    dim: int
    num_layers: int
    num_heads: int

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.dim)
        self.transformer_blocks = [TransformerBlock(num_heads=self.num_heads) for _ in range(self.num_layers)]

    def __call__(self, input_ids):
        x = self.embedding(input_ids)
        for block in self.transformer_blocks:
            x = block(x)
        return x


### src/utils.py
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

def create_train_state(rng, model, learning_rate, input_shape):
    params = model.init(rng, jnp.ones(input_shape))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def cross_entropy_loss(logits, labels):
    logits = logits.reshape(-1, logits.shape[-1])
    labels = labels.reshape(-1)
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits, one_hot).mean()
    return loss

def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['input_ids'])
        loss = cross_entropy_loss(logits, batch['labels'])
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def generate_text(state, tokenizer, prompt, max_length=50):
    input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]
    input_ids = jnp.array(input_ids)
    
    for _ in range(max_length):
        logits = state.apply_fn({'params': state.params}, input_ids)
        next_token_logits = logits[:, -1, :]
        next_token = jnp.argmax(next_token_logits, axis=-1)
        input_ids = jnp.concatenate([input_ids, next_token[:, None]], axis=1)

    return tokenizer.decode(input_ids[0])


### src/train.py
from src.data import load_and_preprocess_data
from src.model import MiniGPT
from src.utils import create_train_state, train_step, generate_text
import jax


def train_model(epochs=5, batch_size=32):
    dataset, tokenizer = load_and_preprocess_data(batch_size=batch_size)
    rng = jax.random.PRNGKey(0)

    model = MiniGPT(vocab_size=tokenizer.vocab_size, dim=256, num_layers=4, num_heads=8)
    state = create_train_state(rng, model, 1e-3, (1, 128))

    for epoch in range(epochs):
        total_loss = 0
        total_samples = 0

        for batch in dataset:
            if "input_ids" not in batch or len(batch["input_ids"]) == 0:
                continue

            batch_size = len(batch["input_ids"])
            state, loss = train_step(state, batch)
            total_loss += loss * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    prompt = "JAX and Flax are powerful tools for"
    generated_text = generate_text(state, tokenizer, prompt)

    return {
        "model_initialized": True,
        "final_loss": float(avg_loss),
        "generated_text": generated_text
    }
