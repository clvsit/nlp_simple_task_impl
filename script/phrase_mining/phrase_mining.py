import json
import jieba
import numpy as np
from queue import Queue
from gensim.models import Word2Vec


class PhraseMining:
    
    def __init__(self, model_wv=None):
        self.model_wv = model_wv
    
    def find(self, start_words: list, center_words: list = None, neg_words: list = None, min_sim: float = 0.6, max_sim: float = 1.0, alpha: float = 0.25) -> list:
        """
        根据启动的种子词去挖掘新词
        """
        if self.model_wv is None:
            print("The word2vec model is None!")
            return []
        
        # 获取词向量大小
        word_size = self.model_wv.vector_size
        
        if center_words is None and neg_words is None:
            min_sim = max(min_sim, 0.6)

        center_vec, neg_vec = np.zeros([word_size]), np.zeros([word_size])

        if center_words:
            _ = 0
            for w in center_words:
                if w in self.model_wv.wv.vocab:
                    center_vec += self.model_wv[w]
                    _ += 1
            if _ > 0:
                center_vec /= _

        if neg_words:
            _ = 0
            for w in neg_words:
                if w in self.model_wv.wv.vocab:
                    neg_vec += self.model_wv[w]
                    _ += 1
            if _ > 0:
                neg_vec /= _

        queue_count = 1
        task_count = 0
        cluster = []
        queue = Queue()
        
        for w in start_words:
            queue.put((0, w))
            if w not in cluster:
                cluster.append(w)

        while not queue.empty():
            idx, word = queue.get()
            queue_count -= 1
            task_count += 1
            if word not in self.model_wv.wv:
                continue
            sims = self._most_similar(self.model_wv, word, center_vec, neg_vec)
            min_sim_ = min_sim + (max_sim - min_sim) * (1 - np.exp(-alpha * idx))
            if task_count % 10 == 0:
                log = '%s in cluster, %s in queue, %s tasks done, %s min_sim' % (len(cluster), queue_count, task_count, min_sim_)
                print(log)
            for i, j in sims:
                if j >= min_sim_:
                    if i not in cluster:
                        queue.put((idx + 1, i))
                        if i not in cluster:
                            cluster.append(i)
                        queue_count += 1
        return cluster
    
    @staticmethod
    def _most_similar(model_wv, word, center_vec=None, neg_vec=None):
        vec = model_wv[word] + center_vec - neg_vec
        return model_wv.similar_by_vector(vec, topn=200)


if __name__ == "__main__":
    data_word_list = []

    # 读取数据
    with open("content.json", "r", encoding="utf-8") as file:
        dataset = json.load(file)

    # 进行分词
    for data in dataset:
        data_word_list.append(jieba.lcut(data))

    model_wv = Word2Vec(data_word_list, window=5, size=100, min_count=10, sg=0, negative=5, workers=5)
    phrase_model = PhraseMining(model_wv)
    # 若没有 word2vec 模型，则进行训练
    print(phrase_model.find(["尾灯"], min_sim=0.7, alpha=0.5))

    # 存储 word2vec 模型
    phrase_model.model_wv.save("./model/word2vec.model")

    # 加载 word2vec 模型并直接使用
    # model_wv = Word2Vec.load("./model/word2vec.model")
    # phrase_model = PhraseMining(model_wv)
    # print(phrase_model.find(["尾灯"], min_sim=0.7, alpha=0.5))
