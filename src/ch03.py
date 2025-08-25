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

