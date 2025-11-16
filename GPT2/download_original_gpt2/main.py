import tensorflow as tf
import tqdm
from download_original_gpt2.download_weights import download_and_load_gpt2
from sagar_neuron_gpt2.model_configuration.gpt_model import GPTModel
import torch
import numpy as np
import tiktoken


class GPT2Loader:   
    
    def __init__(self, model_size="124M", models_dir="gpt2"):
        self.model_size = model_size
        self.models_dir = models_dir
        self.GPT_CONFIG_124M = {
            "vocab_size": 50257,    # Vocabulary size
            "context_length": 1024, # Context length
            "emb_dim": 768,         # Embedding dimension
            "n_heads": 12,          # Number of attention heads
            "n_layers": 12,         # Number of layers
            "drop_rate": 0.1,       # Dropout rate
            "qkv_bias": False       # Query-Key-Value bias
            }
        self.gpt =None
        self.settings =None 
        self.params =None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
    def load_model(self):
        self.settings, self.params = download_and_load_gpt2(model_size=self.model_size, models_dir=self.models_dir)


   

    
    def check(self, settings, params):
        print("Settings: ", settings)
        print("Parameter dictioanry  keys: ", params.keys())

        print("\n\n")

        print(params['wte'])
        print("Token embedding weight dimensions: ", params['wte'].shape)

    
    #This is used for implemnting the model with the radom weights
    def implement_model(self):
        model_configs =         {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        # Copy the base configuration and update with specific model settings
        model_name = "gpt2-small (124M)"  # Example model name
        NEW_CONFIG = self.GPT_CONFIG_124M.copy()
        NEW_CONFIG.update(model_configs[model_name])
        NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})
        self.gpt = GPTModel(NEW_CONFIG)
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
    def eval_gpt_model(self):
        self.gpt.eval()
    

    def helper_to_override(self,left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(torch.tensor(right))
    
    def override_weights(self):

        gpt = self.gpt
        assign = self.helper_to_override
        params = self.params

        gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
        gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
        
        for b in range(len(params["blocks"])):
            q_w, k_w, v_w = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.weight = assign(
                gpt.trf_blocks[b].att.W_query.weight, q_w.T)
            gpt.trf_blocks[b].att.W_key.weight = assign(
                gpt.trf_blocks[b].att.W_key.weight, k_w.T)
            gpt.trf_blocks[b].att.W_value.weight = assign(
                gpt.trf_blocks[b].att.W_value.weight, v_w.T)

            q_b, k_b, v_b = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.bias = assign(
                gpt.trf_blocks[b].att.W_query.bias, q_b)
            gpt.trf_blocks[b].att.W_key.bias = assign(
                gpt.trf_blocks[b].att.W_key.bias, k_b)
            gpt.trf_blocks[b].att.W_value.bias = assign(
                gpt.trf_blocks[b].att.W_value.bias, v_b)

            gpt.trf_blocks[b].att.out_proj.weight = assign(
                gpt.trf_blocks[b].att.out_proj.weight, 
                params["blocks"][b]["attn"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].att.out_proj.bias = assign(
                gpt.trf_blocks[b].att.out_proj.bias, 
                params["blocks"][b]["attn"]["c_proj"]["b"])

            gpt.trf_blocks[b].ff.layers[0].weight = assign(
                gpt.trf_blocks[b].ff.layers[0].weight, 
                params["blocks"][b]["mlp"]["c_fc"]["w"].T)
            gpt.trf_blocks[b].ff.layers[0].bias = assign(
                gpt.trf_blocks[b].ff.layers[0].bias, 
                params["blocks"][b]["mlp"]["c_fc"]["b"])
            gpt.trf_blocks[b].ff.layers[2].weight = assign(
                gpt.trf_blocks[b].ff.layers[2].weight, 
                params["blocks"][b]["mlp"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].ff.layers[2].bias = assign(
                gpt.trf_blocks[b].ff.layers[2].bias, 
                params["blocks"][b]["mlp"]["c_proj"]["b"])

            gpt.trf_blocks[b].norm1.scale = assign(
                gpt.trf_blocks[b].norm1.scale, 
                params["blocks"][b]["ln_1"]["g"])
            gpt.trf_blocks[b].norm1.shift = assign(
                gpt.trf_blocks[b].norm1.shift, 
                params["blocks"][b]["ln_1"]["b"])
            gpt.trf_blocks[b].norm2.scale = assign(
                gpt.trf_blocks[b].norm2.scale, 
                params["blocks"][b]["ln_2"]["g"])
            gpt.trf_blocks[b].norm2.shift = assign(
                gpt.trf_blocks[b].norm2.shift, 
                params["blocks"][b]["ln_2"]["b"])

        gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
        gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
        gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

    def get_model(self):
        self.override_weights()
        self.gpt.to(self.device)


    def generate(self, model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

        # For-loop is the same as before: Get logits, and only focus on last time step
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:, -1, :]

            # New: Filter logits with top_k sampling
            if top_k is not None:
                # Keep only top_k values
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

            # New: Apply temperature scaling
            if temperature > 0.0:
                logits = logits / temperature

                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            # Otherwise same as before: get idx of the vocab entry with the highest logits value
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

            if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                break

            # Same as before: append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

        return idx
    

    def initialise(self):
        self.load_model()
        self.check(self.settings, self.params)  # Uncomment to check settings and params
        self.implement_model()
        self.eval_gpt_model()
        self.get_model()

        return self.gpt
    
    def text_to_token_ids(self,text, tokenizer):
        encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
        return encoded_tensor

    def token_ids_to_text(self,token_ids, tokenizer):
        flat = token_ids.squeeze(0) # remove batch dimension
        return tokenizer.decode(flat.tolist())
    

    def let_the_model_speak(self, ask):
        torch.manual_seed(123)

        token_ids = self.generate(
            model=model,
            idx=self.text_to_token_ids(ask, self.tokenizer),
            max_new_tokens=15,
            context_size=self.GPT_CONFIG_124M["context_length"],
            top_k=25,
            temperature=1.4
        )

        print("Output text:\n", self.token_ids_to_text(token_ids, self.tokenizer))

    


if __name__ == "__main__":
    gpt2_loader = GPT2Loader(model_size="124M", models_dir="gpt2")
    model = gpt2_loader.initialise()
    print("Model loaded and weights overridden successfully.")

    ask = input("Chat with model:\nYou: ")
    gpt2_loader.let_the_model_speak(ask)




    

        



    

    
    
    
    


        
        
