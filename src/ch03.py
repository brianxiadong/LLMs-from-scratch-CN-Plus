import torch

inputs = torch.tensor([[0.43, 0.15, 0.89],  # Your     (x^1)
                       [0.55, 0.87, 0.66],  # journey  (x^2)
                       [0.57, 0.85, 0.64],  # starts   (x^3)
                       [0.22, 0.58, 0.33],  # with     (x^4)
                       [0.77, 0.25, 0.10],  # one      (x^5)
                       [0.05, 0.80, 0.55]]  # step     (x^6)
                      )

query = inputs[1]

print(query)
print(inputs.shape[0])
attn_scores_2 = torch.empty(inputs.shape[0])
print(attn_scores_2)

for i,x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

print(attn_scores_2)

res = 0.

for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]

print(res)
print(torch.dot(inputs[0], query))

# 归一化
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# softmax归一化
def softmax_native(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_native = softmax_native(attn_scores_2)
print("Attention weights:", attn_weights_2_native)
print("Sum:", attn_weights_2_native.sum())

attn_weights_2_native = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2_native)
print("Sum:", attn_weights_2_native.sum())

from typing import Any

x_2: Any = inputs[1]
print(x_2)
d_in: Any = inputs.shape[1]
d_out = 2

torch.manual_seed(123)
W_query:Any = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key:Any = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value:Any = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2: Any = x_2 @ W_query
key_2: Any = x_2 @ W_key
value_2: Any = x_2 @ W_value

print(query_2)

keys : Any = inputs @ W_key
values : Any = inputs @ W_value

print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim=-1)
print(attn_weights_2)

context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

import torch.nn as nn
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        self.W_key = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        self.W_value = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T  # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))


batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape) # 2 inputs with 6 tokens each, and each token has embedding dimension 3

class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        # For inputs where `num_tokens` exceeds `context_length`, this will result in errors
        # in the mask creation further below.
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs
        # do not exceed `context_length` before reaching this forward method.
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # Changed transpose
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights) # New

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)

context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)

context_vecs = ca(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)