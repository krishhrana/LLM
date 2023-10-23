from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils import count_params

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
Dimension symbols:
    B - batch size
    S - sequence length
    D - hidden dimension (n_embd)
    H - number of attention heads (n_head)
    HD - hidden dimension of a single attention head (d // n_head)
    V - size of the vocabulary
"""


class MultiHeadAttention(nn.Module):
    """The multi-head attention module in a decoder block."""

    def __init__(self, n_embd: int, n_head: int, p_dropout: float = 0.1):
        super().__init__()
        """Initialize the modules used by multi-head attention."""

        self.n_head = n_head
        attn_hidden_dim = n_embd // n_head

        self.q_attn = nn.Linear(n_embd, n_embd)
        self.k_attn = nn.Linear(n_embd, n_embd)
        self.v_attn = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(p_dropout)

        scale_factor = 1 / torch.sqrt(torch.tensor(attn_hidden_dim))
        self.register_buffer("scale_factor", scale_factor)

    def q_kT_v(self, x: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Project the hidden states to q, kT, v prior to computing attention.

        Args:
            x: embeddings or hidden states (B x S x D) from the previous decoder block

        Returns:
            q: The query vector used by multi-head attention (B x H x S x HD)
            kT: The transpose of the key vector used by multi-head attention (B x H x HD x S)
            v: The value vector used by multi-head attention (B x H x S x HD)
        """
        B, T, C = x.shape

        q = self.q_attn(x)
        kT = self.k_attn(x)
        v = self.v_attn(x)

        q = q.view(B, T, self.n_head, C // self.n_head).permute(0, 2, 1, 3) #(B, nh, T, mini_head)
        kT = kT.view(B, T, self.n_head, C // self.n_head).permute(0, 2, 1, 3).transpose(-1, -2) #(B, nh, mini_head, T)
        v = v.view(B, T, self.n_head, C // self.n_head).permute(0, 2, 1, 3) #(B, nh, T, mini_head)

        return q, kT, v

    def self_attention(self, q: torch.FloatTensor, kT: torch.FloatTensor, v: torch.FloatTensor, attention_mask: Union[torch.FloatTensor, None] = None) -> torch.FloatTensor:
        B, nh, T, mini_head_size = q.shape

        # Creating Mask
        causal_mask = torch.tril(torch.ones(T, T)).to(device)
        if attention_mask is None:
            mask = causal_mask
        else:
            attention_mask = attention_mask.unsqueeze(1)
            mask = torch.multiply(causal_mask, attention_mask)
            mask = mask.unsqueeze(1)

        # affinities
        affinities = (q @ kT) * self.scale_factor

        # Applying Mask
        float_min = torch.finfo(q.dtype).min
        attn_logits = affinities.masked_fill(mask == 0, float_min)

        # Converting to probabilities, Getting Attention, (No Attention Dropout)
        attn_weights = F.softmax(attn_logits, dim = -1)
        attn = attn_weights @ v # (B, nh, T, mini_head)

        # Merging mini heads ---> (B, T, nh*mini_head)
        attn = attn.permute(0, 2, 1, 3).reshape(B, T, nh * mini_head_size) # (B, T, nh*v_dim)
        return attn

    def projection(self, attn: torch.FloatTensor) -> torch.FloatTensor:
        """Apply a dropout and a linear projection to outputs of attention"""
        return self.dropout(self.proj(attn))

    def forward(
        self, x: torch.FloatTensor, attention_mask: Union[torch.FloatTensor, None] = None
    ) -> torch.FloatTensor:
        """A full forward pass of the multi-head attention module.

        Args:
            x: embeddings or hidden states (B x S x D) from the previous decoder block

        Returns:
            y: outputs (B x S x D) of the multi-head attention module
        """

        # Query, Key, Value
        q, kT, v = self.q_kT_v(x)

        # Weighted values
        attn_values = self.self_attention(q=q, kT=kT, v=v, attention_mask=attention_mask)

        # Multiplying by Wo
        y = self.projection(attn_values)
        return y


class FeedForward(nn.Module):
    """The feedforward attention module in a decoder block."""

    def __init__(self, n_embd: int, p_dropout: float = 0.1):
        """Initialize the modules used by feedforward."""
        super().__init__()

        middle_dim = 4 * n_embd  # stick to what GPT-2 does
        self.linear_in = nn.Linear(n_embd, middle_dim)
        self.linear_out = nn.Linear(middle_dim, n_embd)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """A full forward pass of the feedforward module.

        Args:
            x: outputs (B x S x D) of the first Add & Norm operation

        Returns:
            z: outputs (B x S x D) of the feedforward module

        Different from what you saw in class which uses ReLU as the activation,
        we are going to follow GPT-2 which uses GeLU. You should also apply
        self.dropout to the output.
        """

        x = self.linear_in(x)
        x = F.gelu(x)
        x = self.linear_out(x)
        x = self.dropout(x)
        return x


class DecoderBlock(nn.Module):
    """A single decoder block in a decoder language model."""

    def __init__(self, n_embd: int, n_head: int):
        """Initialize the modules used in a decoder block."""
        super().__init__()

        self.ln_1 = nn.LayerNorm(n_embd)
        self.mha = MultiHeadAttention(n_embd, n_head)
        self.ff = FeedForward(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)

    def forward(
        self, x: torch.FloatTensor, attention_mask: Union[torch.FloatTensor, None]
    ) -> torch.FloatTensor:
        """A full forward pass of the decoder block.

        Args:
            x: embeddings or hidden states (B x S x D) from the previous decoder block
            attention_mask (optional): Mask indicating tokens that shouldn't
              be included in self-attention (B x S). 1 stands for a token that is
              included, and 0 stands for a token that isn't.
        Returns:
            y: outputs of the current decoder block

        Different from what you saw in class which uses ReLU as the activation,
        we are going to follow GPT-2 which uses GeLU. You should also apply
        self.dropout to the output.

        A note on where to place layer normalization (LN): in the lecture, you
        saw "post-LN", which applies LN to the outputs of MHA / FF modules after
        the residual is added. Another approach to do this is "pre-LN", which
        appiles LN to the inputs of the attention and feedforward modules. Both
        implementations should pass the tests. See explanations here:
        https://sh-tsang.medium.com/review-pre-ln-transformer-on-layer-normalization-in-the-transformer-architecture-b6c91a89e9ab
        """

        # Pre-LayerNorm
        x = x + self.mha(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))

        return x


class DecoderLM(nn.Module):
    """The decoder language model."""

    def __init__(
        self,
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        n_layer: int,
        p_dropout: float = 0.1,
    ):
        super().__init__()

        self.n_vocab = n_vocab
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_positions = n_positions
        self.n_layer = n_layer
        self.p_dropout = p_dropout

        self.token_embeddings = nn.Embedding(n_vocab, n_embd)
        self.position_embeddings = nn.Embedding(n_positions, n_embd)
        self.blocks = nn.ModuleList(
            [DecoderBlock(n_embd, n_head) for _ in range(n_layer)]
        )
        self.ln = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(self.p_dropout)

        # initialize weights according to nanoGPT
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / torch.sqrt(2 * n_layer))

        # count flops per token according to nanoGPT
        self.flops_per_token = (
            6 * count_params(self) + 12 * n_layer * n_embd * n_positions
        )

    def embed(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Union[torch.FloatTensor, None] = None,
    ) -> torch.FloatTensor:
        """Convert input_ids to embeddings (token_embeddings + positional_embeddings).

        Args:
            input_ids: tokens ids with shape (B x S)

        Returns:
            embeddings: token representations with shape (B x S x D)
            attention_mask (optional): Mask indicating whether tokens should be
              ignored.
        """

        """
        Position ids are indices of tokens in the sequence. When attention_mask
        isn't provided, they are simply [0, 1, 2, ...] for every sequence in the
        batch. When they are provided, you should ignore tokens with attention_mask
        equal to 0.
        
        Example (B = 2, S = 5):
        
        attention_mask = tensor([
         [0., 0., 1., 1., 1.],
         [1., 1., 1., 1., 1.]
        ])

        position_ids = tensor([
         [0, 0, 0, 1, 2],
         [0, 1, 2, 3, 4]
        ])

        Note that the position ids for masked out tokens do not matter, as long
        as they don't trigger out-of-bounds errors when fed into the embedding
        layer. I.e., they should be within [0, n_positions).

        Hint: torch.cumsum
        """

        assert input_ids.shape[1] <= self.n_positions
        B, T = input_ids.shape
        token_embeddings = self.token_embeddings(input_ids) # (B, T, n_embed)
        if attention_mask is None:
            positional_embeddings = self.position_embeddings(torch.arange(T, dtype = torch.long, device = device)) # (T, n_embed)
        else:
            masked_positions = (torch.cumsum(attention_mask, dim = 1) - 1).masked_fill(attention_mask == 0, 0) # (B, T)
            positional_embeddings = self.position_embeddings(masked_positions.to(torch.long)) #(B, T, n_embed)
        return self.dropout(token_embeddings + positional_embeddings)

    def token_logits(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Project the final hidden states of the model to token logits.

        Args:
            x: hidden states produced by the final decoder block (B x S x D)

        Returns:
            logits: logits corresponding to the predicted next token likelihoods (B x S x V)

        Hint: Question 1.2.
        """
        logits = x @ self.token_embeddings.weight.transpose(-1, -2)
        return logits

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Union[torch.FloatTensor, None] = None,
    ) -> torch.FloatTensor:
        """A forward pass of the decoder LM, converting input_ids to token logits.

        Args:
            input_ids: tokens ids with shape (B x S)
            attention_mask (optional): Mask indicating whether tokens should be
              ignored.

        Returns:
            logits: logits corresponding to the predicted next token likelihoods (B x S x V)
        """
        x = self.embed(input_ids=input_ids, attention_mask=attention_mask)
        for decoder_blocks in self.blocks:
            x = decoder_blocks.forward(x, attention_mask=attention_mask)
        x = self.ln(x)
        logits = self.token_logits(x)
        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not ...:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
