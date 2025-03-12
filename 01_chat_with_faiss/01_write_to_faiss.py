import faiss
import pandas as pd
import os
import numpy as np
from llama_index.legacy.vector_stores.faiss import FaissVectorStore
from base.models import *
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)


###############################################
# 将知识库数据embedding化，写入到faiss向量库
# 1. documents->embeddings
# 2. embeddings to faiss
###############################################


# 配置初始化
API_KEY = os.getenv('API_KEY')

# 流式读取数据，每次读取1行
chunk_size = 1
embeddings = []
texts = []
nodes = []

embed_model = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
    api_key=API_KEY
)

# 从文件里依次读取每行文本
for chunk in pd.read_csv('../data/运动鞋店铺知识库.txt', sep='\t', names=['passage'], chunksize=chunk_size):
    batch_text = chunk['passage'].values[0]
    # 打印读取的单行文本
    print(batch_text)
    print("[INFO]batch_text print end============================")

    try:
        # TODO 参考API文档 https://help.aliyun.com/zh/model-studio/developer-reference/dashscopeembedding-in-llamaindex
        # 调用 embed_model 模型API,返回每行文本对应的向量
        batch_embedding = [[]]
        print(batch_embedding)
        print("[INFO]batch_embeddings print end============================")
    except Exception as e:
        print(f"Error in embedding generation: {e}")
        continue

    texts.append(batch_text)
    embeddings.append(batch_embedding[0])
    nodes.append(CustomNode(text=batch_text, embedding=batch_embedding[0]))

# 输出向量节点
print(nodes)
print("[INFO]nodes print end============================")

# 生成 embeddings 并创建 FAISS 索引
embeddings_np = np.array(embeddings).astype('float32')
# 初始化faiss，embedding索引长度embeddings_np.shape[1]=1536
print(f"[INFO]embedding index is {embeddings_np.shape[1]}")

# IndexFlat：最基础的索引类型，直接存储所有向量并进行精确的搜索，适合小规模数据集或对精度要求非常高的场景。
# IndexFlatL2 是一种基于欧式距离（L2 距离）的索引类型，用于精确地进行向量的最近邻搜索，适合小规模数据集或对精度要求较高的场景。
# IndexAnnoy 基于树的近似最近邻搜索方法（Annoy），适合中等规模的数据集，查询速度较快，适用于内存和速度的平衡需求。
# IndexIVFPQ：将倒排文件（IVF）和产品量化（PQ）结合的索引类型，通过量化压缩向量并对簇内数据进行检索，适合大规模数据集且对内存和查询速度有双重要求。
faiss_index = faiss.IndexFlatL2(embeddings_np.shape[1])

# 初始化并写入 FAISS 向量存储
faiss_store = FaissVectorStore(faiss_index=faiss_index)
# 添加节点到 FAISS 向量存储
faiss_store.add(nodes)  # 直接添加节点
# 保存 FAISS 索引到文件
faiss_store.persist(persist_path='../output/faiss_index_test_shop.index')
print("============write index successfully")
