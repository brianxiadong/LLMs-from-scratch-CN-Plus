import os.path
import re

with open("the-verdict.txt" , "r" , encoding="utf-8") as f :
    book_text = f.read()

print("Length of text: ", len(book_text))
print(book_text[:99])

text = "Hello, world. This, is a test."
result = re.split(r'(\s)',text)
print(result)

result = [i for i in result if i.strip()]
print(result)

result2 = re.split(r'([,.]|\s)',text)
print(result2)

result2 = [i for i in result2 if i.strip()]
print(result2)

text = "Hello, world. Is this-- a test?"

result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

book_text = re.split(r'([,.:;?_!"()\']|--|\s)', book_text)
book_text = [item.strip() for item in book_text if item.strip()]
print(book_text)
print(len(book_text))

all_words = sorted(set(book_text))
print(len(all_words))

vocab = {word: i for i, word in enumerate(all_words)}
print(vocab)

# 打印前50条
for i in range(50):
    print(all_words[i])

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: word for word, i in vocab.items()}

    def encoding(self,text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return [self.str_to_int[word] for word in preprocessed]

    def decoding(self, tokens):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# tokenizers = SimpleTokenizerV1(vocab)
# print(tokenizers.encoding("Hello, world. This, is a test."))
# print(tokenizers.decoding([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, ]))

all_tokens = sorted(list(set(book_text)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {word: i for i, word in enumerate(all_tokens)}
print(len(vocab.items()))

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: word for word, i in vocab.items()}

    def encoding(self,text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # 如果单词不存在，就使用"<|unk|>"
        preprocessed = [word if word in self.str_to_int else "<|unk|>" for word in preprocessed]

        return [self.str_to_int[word] for word in preprocessed]

    def decoding(self, tokens):
        text = " ".join([self.int_to_str[i] for i in id])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

tokenizers = SimpleTokenizerV2(vocab)
print(tokenizers.encoding("Hello, world. This, is a test."))

import tiktoken
# 打印tiktoken所有编码方式
print(tiktoken.list_encoding_names())
tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(ids)


with open("the-verdict.txt" , "r" , encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print("Length of text: ", len(enc_text))

enc_sample = enc_text[50:]
context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

import torch

from torch import Dataset, DataLoader

# class GPTDatasetV1(Dataset):
#     def __init__(self , txt , tokenizer , max_length , stride):
#
#
