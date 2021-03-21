# 短语挖掘
对苏神苏剑林大佬的代码进行了调整，仅用来做短语的相似挖掘，《分享一次专业领域词汇的无监督挖掘》：https://spaces.ac.cn/archives/6540 。
- 具体代码：phrase_mining.py
- 使用说明：

第一次使用：

```python
if __name__ == "__main__":
    data_word_list = []

    # 读取数据
    with open("content.json", "r", encoding="utf-8") as file:
        dataset = json.load(file)

    # 进行分词
    for data in dataset:
        data_word_list.append(jieba.lcut(data))

    phrase_model = PhraseMining()
    # 若没有 word2vec 模型，则进行训练
    phrase_model.train(data_word_list)
    print(phrase_model.find(["尾灯"], min_sim=0.7, alpha=0.5))

    # 存储 word2vec 模型
    phrase_model.model_wv.save("./model/word2vec.model")

    # 加载 word2vec 模型并直接使用
    # model_wv = Word2Vec.load("./model/word2vec.model")
    # phrase_model = PhraseMining(model_wv)
    # print(phrase_model.find(["尾灯"], min_sim=0.7, alpha=0.5))
```

加载已有的 word2vec 模型：
```python
if __name__ == "__main__":
    data_word_list = []

    # 读取数据
    with open("content.json", "r", encoding="utf-8") as file:
        dataset = json.load(file)

    # 进行分词
    for data in dataset:
        data_word_list.append(jieba.lcut(data))

    # 加载 word2vec 模型并直接使用
    model_wv = Word2Vec.load("./model/word2vec.model")
    phrase_model = PhraseMining(model_wv)
    print(phrase_model.find(["尾灯"], min_sim=0.7, alpha=0.5))
```

## 函数介绍
- find()：

参数名称 | 中文含义 | 数据类型 | 是否必填 | 默认值 | 用法描述
---|---|---|---|---|---
start_words | 启动词 | list | 是 | 无 | 用来做短语挖掘的启动词
center_words | 中心词 | list | 否 | None | 挖掘过程中的中心词
neg_words | 否定词 | list | 否 | None | 挖掘过程中的否定词
min_sim | 最小相似度阈值 | float | 否 | 0.6 | 挖掘过程中的最小相似度阈值 
max_sim | 最大相似度半径 | float | 否 | 1.0 | 达到最大相似度阈值时停止挖掘
alpha | 相似度阈值增加系数 | float | 否 | 0.25 | 每轮挖掘过程相似度阈值提高的系数

每轮最小相似度阈值的计算公式，其中 idx 表示迭代的轮数：
```python
min_sim_ = min_sim + (max_sim - min_sim) * (1 - np.exp(-alpha * idx))
```

## 使用介绍
在这介绍中心词、否定词对短语挖掘效果的影响。

```python
print(phrase_model.find(["尾灯"], min_sim=0.7, alpha=0.5))
# ['尾灯', '前大灯', '贯穿', '灯组', '前灯', '大灯', '内部结构', '车尾', '头灯', '扁平', '两侧', '细长', '羽式', '箭', '灯带', '狭长的', '进气口', '日行', '下部', '尾部', '光源']

print(phrase_model.find(["尾灯"], ["大灯"], min_sim=0.7, alpha=0.5))
# ['尾灯', '大灯', '前大灯', '灯组', '头灯', '前灯', '扁平', '光源', '组', '内部结构', '灯带', '细长', '狭长', 'LED']

print(phrase_model.find(["尾灯"], ["大灯"], ["细长"], min_sim=0.7, alpha=0.5))
# ['尾灯', '大灯', '前大灯', '头灯', '灯组', '光源', '前灯']
```