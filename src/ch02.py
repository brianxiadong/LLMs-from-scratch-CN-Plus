# 导入操作系统路径模块
import os.path
# 导入正则表达式模块
import re

# 打开并读取文本文件"the-verdict.txt"
with open("the-verdict.txt" , "r" , encoding="utf-8") as f :
    book_text = f.read()  # 读取整个文件内容到book_text变量

# 打印文本长度
print("Length of text: ", len(book_text))
# 打印文本的前99个字符
print(book_text[:99])

# 定义一个测试文本
text = "Hello, world. This, is a test."
# 使用正则表达式按空格分割文本，保留分隔符
result = re.split(r'(\s)',text)
# 打印分割结果
print(result)

# 过滤掉空白字符，只保留非空的元素
result = [i for i in result if i.strip()]
# 打印过滤后的结果
print(result)

# 使用更复杂的正则表达式分割，包括逗号、句号和空格
result2 = re.split(r'([,.]|\s)',text)
# 打印分割结果
print(result2)

# 过滤掉空白字符
result2 = [i for i in result2 if i.strip()]
# 打印过滤后的结果
print(result2)

# 定义一个包含更多标点符号的测试文本
text = "Hello, world. Is this-- a test?"

# 使用完整的正则表达式分割文本，包括各种标点符号
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
# 去除空白并过滤空字符串
result = [item.strip() for item in result if item.strip()]
# 打印结果
print(result)

# 对整本书的文本进行同样的分割处理
book_text = re.split(r'([,.:;?_!"()\']|--|\s)', book_text)
# 去除空白并过滤空字符串
book_text = [item.strip() for item in book_text if item.strip()]
# 打印处理后的文本
print(book_text)
# 打印分割后的总长度
print(len(book_text))

# 创建唯一单词的排序列表
all_words = sorted(set(book_text))
# 打印唯一单词的数量
print(len(all_words))

# 创建词汇表字典，将单词映射到索引
vocab = {word: i for i, word in enumerate(all_words)}
# 打印词汇表
print(vocab)

# 打印前50个单词
for i in range(50):
    print(all_words[i])  # 打印第i个单词

# 定义简单的分词器版本1
class SimpleTokenizerV1:
    def __init__(self, vocab):
        # 初始化分词器，接收词汇表作为参数
        self.str_to_int = vocab  # 字符串到整数的映射
        self.int_to_str = {i: word for word, i in vocab.items()}  # 整数到字符串的映射

    def encoding(self,text):
        # 编码方法：将文本转换为token ID列表
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)  # 使用正则表达式分割文本
        preprocessed = [item.strip() for item in preprocessed if item.strip()]  # 去除空白字符
        return [self.str_to_int[word] for word in preprocessed]  # 将每个单词转换为对应的ID

    def decoding(self, tokens):
        # 解码方法：将token ID列表转换回文本
        text = " ".join([self.int_to_str[i] for i in tokens])  # 修复：将ids改为tokens
        # 去除指定标点符号前的空格
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text  # 返回解码后的文本

# 创建SimpleTokenizerV1实例并测试（注释掉的代码）
# tokenizers = SimpleTokenizerV1(vocab)
# print(tokenizers.encoding("Hello, world. This, is a test."))  # 测试编码功能
# print(tokenizers.decoding([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, ]))  # 测试解码功能

# 创建包含特殊token的词汇表
all_tokens = sorted(list(set(book_text)))  # 获取所有唯一的token并排序
all_tokens.extend(["<|endoftext|>", "<|unk|>"])  # 添加特殊token：文本结束标记和未知词标记
vocab = {word: i for i, word in enumerate(all_tokens)}  # 创建新的词汇表映射
print(len(vocab.items()))  # 打印词汇表的大小

# 定义改进的分词器版本2，支持未知词处理
class SimpleTokenizerV2:
    def __init__(self, vocab):
        # 初始化分词器，接收词汇表作为参数
        self.str_to_int = vocab  # 字符串到整数的映射
        self.int_to_str = {i: word for word, i in vocab.items()}  # 整数到字符串的映射

    def encoding(self,text):
        # 编码方法：将文本转换为token ID列表，支持未知词处理
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)  # 使用正则表达式分割文本
        preprocessed = [item.strip() for item in preprocessed if item.strip()]  # 去除空白字符

        # 如果单词不存在于词汇表中，就使用"<|unk|>"替换
        preprocessed = [word if word in self.str_to_int else "<|unk|>" for word in preprocessed]

        return [self.str_to_int[word] for word in preprocessed]  # 将每个单词转换为对应的ID

    def decoding(self, tokens):
        # 解码方法：将token ID列表转换回文本
        text = " ".join([self.int_to_str[i] for i in tokens])  # 修复：将id改为tokens
        # 去除指定标点符号前的空格
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text  # 返回解码后的文本

# 创建SimpleTokenizerV2实例并测试
tokenizers = SimpleTokenizerV2(vocab)  # 实例化改进的分词器
print(tokenizers.encoding("Hello, do you like tea?"))  # 测试编码功能
print(tokenizers.decoding(tokenizers.encoding("Hello, do you like tea?")))  # 测试解码功能

# 使用tiktoken库进行分词（OpenAI的官方分词器）
import tiktoken  # 导入tiktoken库
tokenizer = tiktoken.get_encoding("gpt2")  # 获取GPT-2的分词器
text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."  # 测试文本
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})  # 编码文本，允许特殊token
print(integers)  # 打印编码结果
strings = tokenizer.decode(integers)  # 解码回文本
print(strings)  # 打印解码结果

# 导入PyTorch相关模块
import torch  # 导入PyTorch主模块
from torch.utils.data import Dataset, DataLoader  # 修复：从torch.utils.data导入

# 定义GPT数据集类，用于处理文本数据
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        # 初始化数据集
        self.input_ids = []  # 存储输入序列
        self.target_ids = []  # 存储目标序列

        # 对整个文本进行分词
        token_ids = tokenizer.encode(txt)

        # 使用滑动窗口将文本分割成重叠的max_length长度序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]  # 输入序列
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 目标序列（向右偏移1位）
            self.input_ids.append(torch.tensor(input_chunk))  # 转换为tensor并添加到列表
            self.target_ids.append(torch.tensor(target_chunk))  # 转换为tensor并添加到列表

    def __len__(self):
        # 返回数据集的长度
        return len(self.input_ids)

    def __getitem__(self, idx):
        # 根据索引返回数据项
        return self.input_ids[idx], self.target_ids[idx]


# 重新读取原始文本用于数据加载器
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()  # 读取原始文本数据

# 设置模型参数
vocab_size = 50257  # 词汇表大小
output_dim = 256  # 输出维度
max_length = 1024  # 最大序列长度
# 创建数据加载器，用于批量加载训练数据
dataloader = DataLoader(GPTDatasetV1(raw_text, tokenizer, max_length, 1), batch_size=8, shuffle=True, drop_last=True)

# 测试数据加载器
data_iter = iter(dataloader)  # 创建数据迭代器
inputs, targets = next(data_iter)  # 获取一个批次的数据
print("Inputs:\n", inputs)  # 打印输入数据
print("\nTargets:\n", targets)  # 打印目标数据


