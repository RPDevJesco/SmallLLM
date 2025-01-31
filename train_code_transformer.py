import os
import numpy as np
from tqdm import tqdm

# Configuration
DATA_PATH = r"E:\Software Dev\Test Repositoriues"  # Raw string for Windows paths
FILE_EXTENSIONS = [".py", ".cs", ".js", ".java", ".ts", ".html", ".css", ".md", ".go", ".rs", ".jsx", ".ml", ".tsx", ".rb", ".kt" ]
WINDOW_SIZE = 32  # Reduced window size for better learning
EMBED_DIM = 64    # Increased embedding size
BATCH_SIZE = 32   # Increased batch size
EPOCHS = 20
LEARNING_RATE = 0.0005  # Reduced learning rate

def load_text_data():
    """Load all code files from your repositories"""
    text = ""
    file_count = 0
    for root, _, files in tqdm(os.walk(DATA_PATH), desc="Scanning repositories"):
        for file in files:
            if any(file.endswith(ext) for ext in FILE_EXTENSIONS):
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        content = f.read()
                        if content.strip():  # Skip empty files
                            text += content + "\n"
                            file_count += 1
                except (UnicodeDecodeError, Exception) as e:
                    continue
    print(f"Loaded {file_count} files")
    return text

def preprocess_data(text):
    """Create character-level vocabulary with special tokens"""
    # Add special tokens
    SPECIAL_TOKENS = ['<PAD>', '<UNK>']
    chars = SPECIAL_TOKENS + sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    # Convert text to indices
    encoded_text = [char_to_idx.get(ch, char_to_idx['<UNK>']) for ch in text]
    return encoded_text, char_to_idx, idx_to_char

class SlidingWindowTransformer:
    def __init__(self, vocab_size, window_size, embed_dim):
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Initialize weights with proper scaling
        scale = np.sqrt(2.0 / (vocab_size + embed_dim))
        self.char_embed = np.random.randn(vocab_size, embed_dim) * scale
        self.pos_embed = np.random.randn(window_size, embed_dim) * scale

        # Transformer components with proper initialization
        self.W_Q = np.random.randn(embed_dim, embed_dim) * scale
        self.W_K = np.random.randn(embed_dim, embed_dim) * scale
        self.W_V = np.random.randn(embed_dim, embed_dim) * scale
        self.W_ff = np.random.randn(embed_dim, embed_dim) * scale
        self.W_out = np.random.randn(embed_dim, vocab_size) * scale

        # Layer normalization parameters
        self.gamma1 = np.ones(embed_dim)
        self.beta1 = np.zeros(embed_dim)
        self.gamma2 = np.ones(embed_dim)
        self.beta2 = np.zeros(embed_dim)

    def layer_norm(self, x, gamma, beta):
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + 1e-5) + beta

    def forward(self, x):
        # Embedding layer
        char_emb = self.char_embed[x]
        pos_emb = self.pos_embed[:len(x)]
        x_emb = char_emb + pos_emb

        # Multi-head attention
        Q = np.dot(x_emb, self.W_Q)
        K = np.dot(x_emb, self.W_K)
        V = np.dot(x_emb, self.W_V)

        # Scaled dot-product attention
        scale = np.sqrt(self.embed_dim)
        attn_scores = np.dot(Q, K.T) / scale

        # Apply causal mask
        mask = np.triu(np.ones_like(attn_scores), k=1)
        attn_scores = np.ma.masked_array(attn_scores, mask=mask, fill_value=-1e9)

        # Softmax and attention
        attn_weights = np.exp(attn_scores) / np.sum(np.exp(attn_scores), axis=-1, keepdims=True)
        attn_out = np.dot(attn_weights, V)

        # First residual connection and layer norm
        attn_out = self.layer_norm(x_emb + attn_out, self.gamma1, self.beta1)

        # Feedforward network
        ff_out = np.tanh(np.dot(attn_out, self.W_ff))

        # Second residual connection and layer norm
        ff_out = self.layer_norm(attn_out + ff_out, self.gamma2, self.beta2)

        # Final output projection
        logits = np.dot(ff_out[-1], self.W_out)

        return logits, (x_emb, attn_weights, attn_out, ff_out, Q, K, V)

    def backward(self, x, logits, cache, target, lr):
        x_emb, attn_weights, attn_out, ff_out, Q, K, V = cache
        window_size, embed_dim = x_emb.shape

        # Gradient clipping value
        clip_value = 5.0

        # Compute softmax gradients
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)
        d_logits = probs.copy()
        d_logits[target] -= 1
        np.clip(d_logits, -clip_value, clip_value, out=d_logits)

        # Backprop through output layer
        d_W_out = np.outer(ff_out[-1], d_logits)
        d_ff = np.zeros_like(ff_out)
        d_ff[-1] = np.dot(self.W_out, d_logits)
        np.clip(d_ff, -clip_value, clip_value, out=d_ff)

        # Backprop through second layer norm
        d_ff_norm = d_ff.copy()
        d_gamma2 = np.sum(d_ff_norm * (ff_out - np.mean(ff_out, axis=-1, keepdims=True))
                          / np.sqrt(np.var(ff_out, axis=-1, keepdims=True) + 1e-5), axis=0)
        d_beta2 = np.sum(d_ff_norm, axis=0)

        # Backprop through feedforward
        d_ff_pre = (1 - ff_out**2) * np.dot(d_ff_norm, self.W_ff.T)
        d_W_ff = np.dot(attn_out.T, d_ff_pre)

        # Backprop through first residual connection
        d_attn = d_ff_pre + d_ff  # Add residual gradient

        # Backprop through first layer norm
        d_attn_norm = d_attn.copy()
        d_gamma1 = np.sum(d_attn_norm * (attn_out - np.mean(attn_out, axis=-1, keepdims=True))
                          / np.sqrt(np.var(attn_out, axis=-1, keepdims=True) + 1e-5), axis=0)
        d_beta1 = np.sum(d_attn_norm, axis=0)

        # Attention backward pass
        scale = np.sqrt(embed_dim)

        # Gradient for V
        d_V = np.dot(attn_weights.T, d_attn_norm)  # (window_size, embed_dim)

        # Gradient for attention weights with causal mask
        d_attn_weights = np.dot(d_attn_norm, V.T) / scale  # (window_size, window_size)
        mask = np.triu(np.ones_like(d_attn_weights), k=1)
        d_attn_weights = np.ma.masked_array(d_attn_weights, mask=mask, fill_value=0).filled()

        # Gradients for Q and K
        d_Q = np.dot(d_attn_weights, K) / scale  # (window_size, embed_dim)
        d_K = np.dot(d_attn_weights.T, Q) / scale  # (window_size, embed_dim)

        # Clip attention gradients
        for grad in [d_V, d_Q, d_K]:
            np.clip(grad, -clip_value, clip_value, out=grad)

        # Embedding gradients (including residual connections)
        d_emb = (
                np.dot(d_V, self.W_V.T) +
                np.dot(d_Q, self.W_Q.T) +
                np.dot(d_K, self.W_K.T) +
                d_attn  # Add residual gradient
        )
        np.clip(d_emb, -clip_value, clip_value, out=d_emb)

        # Update parameters
        # Weight matrices
        self.W_out -= lr * d_W_out
        self.W_ff -= lr * d_W_ff
        self.W_V -= lr * np.dot(x_emb.T, d_V)
        self.W_Q -= lr * np.dot(x_emb.T, d_Q)
        self.W_K -= lr * np.dot(x_emb.T, d_K)

        # Layer norm parameters
        self.gamma1 -= lr * d_gamma1
        self.beta1 -= lr * d_beta1
        self.gamma2 -= lr * d_gamma2
        self.beta2 -= lr * d_beta2

        # Update embeddings
        for i, idx in enumerate(x):
            self.char_embed[idx] -= lr * d_emb[i]
            self.pos_embed[i] -= lr * d_emb[i]

def create_dataset(encoded_text, window_size):
    """Create training samples using sliding window"""
    X, Y = [], []
    for i in range(len(encoded_text) - window_size - 1):
        X.append(encoded_text[i:i+window_size])
        Y.append(encoded_text[i+window_size])
    return np.array(X), np.array(Y)

def create_batches(X, Y, batch_size):
    """Generate mini-batches for training"""
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        batch_indices = indices[start:end]
        yield X[batch_indices], Y[batch_indices]

def train():
    # Load and preprocess data
    print("Loading data...")
    raw_text = load_text_data()
    if not raw_text:
        raise ValueError(f"No valid code files found in {DATA_PATH}")

    print("Preprocessing data...")
    encoded_text, char_to_idx, idx_to_char = preprocess_data(raw_text)
    vocab_size = len(char_to_idx)
    print(f"Vocabulary size: {vocab_size}")

    # Create dataset
    X, Y = create_dataset(encoded_text, WINDOW_SIZE)
    print(f"Dataset created with {len(X)} samples")

    # Initialize model
    model = SlidingWindowTransformer(vocab_size, WINDOW_SIZE, EMBED_DIM)

    # Training loop with improved logging
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        total_loss = 0
        batch_count = 0

        for X_batch, Y_batch in tqdm(create_batches(X, Y, BATCH_SIZE),
                                     desc=f"Epoch {epoch+1}/{EPOCHS}"):
            batch_loss = 0
            for x, y in zip(X_batch, Y_batch):
                logits, cache = model.forward(x)
                loss = -np.log(softmax(logits)[y] + 1e-10)  # Add small epsilon
                batch_loss += loss
                model.backward(x, logits, cache, y, LEARNING_RATE)

            avg_batch_loss = batch_loss / len(X_batch)
            total_loss += avg_batch_loss
            batch_count += 1

            # Print intermittent samples
            if batch_count % 100 == 0:
                print(f"\nBatch {batch_count} loss: {avg_batch_loss:.4f}")
                print_sample(model, char_to_idx, idx_to_char)

        epoch_loss = total_loss / batch_count
        print(f"\nEpoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f}")

        # Generate longer sample at epoch end
        print("\nEpoch-end sample:")
        print_sample(model, char_to_idx, idx_to_char, length=200)

        # Save best model (you'll need to implement save_model)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # save_model(model, 'best_model.pkl')

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def generate_text(model, seed, char_to_idx, idx_to_char, window_size, length=100):
    """Generate text from a seed string"""
    generated = [char_to_idx.get(c, 0) for c in seed]
    for _ in range(length):
        context = generated[-window_size:]
        if len(context) < window_size:
            context = [0] * (window_size - len(context)) + context
        logits, _ = model.forward(np.array(context))
        probs = np.exp(logits) / np.sum(np.exp(logits))
        next_char = np.random.choice(len(probs), p=probs)
        generated.append(next_char)
    return ''.join([idx_to_char.get(i, '') for i in generated[-length:]])

def print_sample(model, char_to_idx, idx_to_char, seed="def ", length=100):
    """Generate and print a sample with proper formatting"""
    generated = generate_text(model, seed, char_to_idx, idx_to_char, WINDOW_SIZE, length)
    print(f"Generated text (seed: '{seed}'):")
    print("-" * 40)
    print(generated)
    print("-" * 40)

if __name__ == "__main__":
    train()