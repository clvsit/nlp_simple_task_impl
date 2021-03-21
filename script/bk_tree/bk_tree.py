import tqdm


def edit_distance(str_a: str, str_b: str) -> int:
    m, n = len(str_a), len(str_b)
    dp_table = []

    for row in range(m + 1):
        dp_table.append([0] * (n + 1))

    for i in range(m + 1):
        dp_table[i][0] = i
    for j in range(n + 1):
        dp_table[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str_a[i - 1] == str_b[j - 1]:
                dp_table[i][j] = dp_table[i - 1][j - 1]
            else:
                dp_table[i][j] = min(
                    dp_table[i - 1][j - 1],
                    dp_table[i][j - 1],
                    dp_table[i - 1][j]
                ) + 1

    return dp_table[m][n]


class Node:

    def __init__(self, word: str):
        self.word = word
        self.branch = {}


class BKTree:

    def __init__(self):
        self.root = None
        self.word_list = []

    def build(self, word_list: list) -> None:
        """
        构建 BK 树
        :param word_list: list 词语列表
        :return: None
        """
        if not word_list:
            return None

        self.word_list = word_list

        # 首先，挑选第一个词语作为 BK 树的根结点
        self.root = Node(word_list[0])

        # 然后，依次往 BK 树中插入剩余的词语
        for word in tqdm.tqdm(word_list[1:]):
            self._build(self.root, word)

    def _build(self, parent_node: Node, word: str) -> None:
        """
        具体实现函数：构建 BK 树
        :param parent_node: Node 父节点
        :param word:        str  待添加到 BK 树的词语
        :return: None
        """
        dis = edit_distance(parent_node.word, word)

        # 判断当前距离（边）是否存在，若不存在，则创建新的结点；否则，继续沿着子树往下走
        if dis not in parent_node.branch:
            parent_node.branch[dis] = Node(word)
        else:
            self._build(parent_node.branch[dis], word)

    def query(self, query_word: str, max_dist: int, min_dist: int = 0) -> list:
        """
        BK 树查询
        :param query_word: str 查询词语
        :param max_dist:   int 最大距离
        :param min_dist:   int 最小距离
        :return: list 符合距离范围的词语列表
        """
        result = []

        self._traverse_and_get(query_word, max_dist, min_dist, self.root, result)
        return result

    def _traverse_and_get(self, query_word: str, max_dist: int, min_dist: int, node: Node, result: list) -> None:
        """
        具体实现函数：BK 树查询
        :param query_word: str  查询词语
        :param max_dist:   int  最大距离
        :param min_dist:   int  最小距离
        :param node:       Node 当前节点
        :param result:     list 符合距离范围的词语列表
        :return: None
        """
        if not node:
            return None

        dis = edit_distance(query_word, node.word)

        # 根据三角不等式来确定查询范围，以实现剪枝的目的
        left, right = max(0, dis - max_dist), dis + max_dist

        for dis_range in range(left, right + 1):
            if dis_range in node.branch:
                dis_branch = edit_distance(query_word, node.branch[dis_range].word)

                # 符合距离范围的词语，将其添加到 result 列表中
                if min_dist <= dis_branch <= max_dist:
                    result.append(node.branch[dis_range].word)

                # 继续沿着子节点遍历，直到叶子节点
                self._traverse_and_get(query_word, max_dist, min_dist, node.branch[dis_range], result)

    def traverse_and_print(self, node: Node):
        if not node:
            print(self.root.word)
            self._traverse_and_print(self.root)
        else:
            self._traverse_and_print(node)

    def _traverse_and_print(self, node: Node):
        if node:
            for dis, child in node.branch.items():
                print(dis, child.word)
                self._traverse_and_print(child)


if __name__ == '__main__':
    word_list = ["game", "fame", "same", "gate", "gain", 'gay', "frame", "home", "aim", "acm", "ame", "fell", "fbcdg"]
    bk_tree = BKTree()
    bk_tree.build(word_list)
    query_word = "fame"
    print(bk_tree.query(query_word, 2, min_dist=2))
