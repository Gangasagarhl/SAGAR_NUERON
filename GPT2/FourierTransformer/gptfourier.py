import torch 
from torch import nn
from FourierTransformer.transfomer.transformer_architecture import TransformerBlock
from FourierTransformer.spectral_filter import SpectralFilter
from FourierTransformer.transfomer.layer_norm import LayerNorm

class GPTModelFourier(nn.Module):
    def __init__(self, cfg, retain_ratio=0.5, insert_every=2):
        super().__init__()

        self.cfg = cfg
        self.insert_every = insert_every
        self.retain_ratio = retain_ratio

        # original GPT embeddings
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

       
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Fourier filter
        self.fourier_filter = SpectralFilter(retain_ratio)

        # final output head
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        B, L = in_idx.shape # batch size and sequence length

        # embeddings
        tok = self.tok_emb(in_idx)                              # (B, L, D)
        pos = self.pos_emb(torch.arange(L, device=in_idx.device))  # (L, D)
        x = tok + pos
        x = self.drop_emb(x)

        # run transformer blocks one by one
        for i, block in enumerate(self.trf_blocks):

            x = block(x)                                        # (B, L, D)

            # insert Fourier filter after every N blocks
            if (i + 1) % self.insert_every == 0:
                x = self.fourier_filter(x)                      # (B, L', D)
                L = x.size(1)                                   # update seq length dynamically

        x = self.final_norm(x)                                  # (B, L', D)
        logits = self.out_head(x)                               # (B, L', vocab)
        return logits
