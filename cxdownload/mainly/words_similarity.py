from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# Print a loading message
print("正在加载模型，请稍等...")

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Notify the user that the model is loaded
print("模型加载完成，现在可以输入句子进行相似度对比。")

def calculate_similarity(sentence1, sentence2):
    # 2. Calculate embeddings by calling model.encode()
    embeddings1 = model.encode([sentence1])
    embeddings2 = model.encode([sentence2])

    # Convert numpy arrays to torch tensors
    embeddings1 = torch.from_numpy(np.array(embeddings1))
    embeddings2 = torch.from_numpy(np.array(embeddings2))

    # 3. Calculate the cosine similarity
    cosine_similarity = torch.cosine_similarity(embeddings1, embeddings2, dim=1)
    return cosine_similarity.item()

# Continuous loop to allow multiple inputs
while True:
    # Input sentences
    sentence_a = input("请输入第一句话 (输入'退出'来结束程序): ")
    if sentence_a.lower() == '退出':
        break
    sentence_b = input("请输入第二句话: ")

    # Calculate and print similarity
    similarity = calculate_similarity(sentence_a, sentence_b)
    print(f"{sentence_a}和{sentence_b}这两句话的相似度为: {similarity:.4f}")
