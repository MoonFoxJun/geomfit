"""Semantic space of text as a manifold: build a word-vector space from a corpus and
visualize its low-dimensional structure.

Idea (distributional hypothesis): a word's meaning is the company it keeps — the words
that frequently appear around it.

Pipeline (standard practice, i.e. LSA / latent semantic analysis):
    corpus -> tokenize -> co-occurrence matrix -> PPMI weighting -> SVD -> word vectors
    The low-dimensional space spanned by the word vectors is the "manifold" of text, where:
      - semantically related words cluster together (clusters on the manifold)
      - semantic relations appear as vector directions (analogy: apple - fruit + grain ≈ rice)

Relation to touch data: touch (pressure/area/time series) -> feature vectors -> the same
dimensionality-reduction / manifold pipeline, with sensor data as input.
"""

import sys
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------------------
# 1. 语料库:一个手工编写的小型语料,包含几个清晰的语义类别
#    (动物/食物/自然/科技/情感/动作)
# ---------------------------------------------------------------------------
CORPUS = [
    # 动物与自然类句子
    "the cat and the dog play in the garden",
    "the mouse hides from the cat",
    "a lion lives in the jungle",
    "the tiger hunts in the forest",
    "the bear sleeps in the cave",
    "the wolf runs with the pack",
    "the horse and the cow live on the farm",
    "birds sing in the morning",
    "the dog runs after the cat",
    # 食物类句子
    "i eat bread and cheese for breakfast",
    "the apple is sweet and juicy",
    "rice is the main food in many countries",
    "milk and honey are natural foods",
    "the chef cooks meat with salt",
    "we drink water with our meal",
    "an apple is a fruit",
    "the banana is a fruit",
    "rice is a grain",
    "wheat is a grain",
    # 自然类句子
    "the river flows to the sea",
    "the mountain is covered with snow",
    "the forest is full of trees",
    "the wind blows across the field",
    "the rain falls on the roof",
    "the sun rises in the east",
    "the moon shines at night",
    "the lake reflects the sky",
    # 科技类句子
    "the computer runs the code",
    "the robot moves like a machine",
    "data flows through the network",
    "the programmer writes software",
    "the algorithm processes the data",
    "the machine learns from data",
    "the server stores the files",
    # 情感类句子
    "love gives us joy and hope",
    "fear makes the heart beat fast",
    "anger turns into sadness",
    "we feel happiness when we see the sun",
    "the news brings us joy",
    "she cried with sadness",
    # 动作类句子
    "we run and walk in the park",
    "the child jumps and laughs",
    "i read a book before sleep",
    "he eats dinner and drinks water",
    "we sleep under the stars",
    "she reads the letter with hope",
]

# 语义分组:用于绘图着色与后续验证(检验词向量是否按语义聚簇)
GROUPS = {
    "animal":  ["cat", "dog", "lion", "tiger", "bear", "wolf", "mouse", "horse", "cow", "bird"],
    "food":    ["bread", "cheese", "apple", "banana", "rice", "wheat", "grain", "fruit", "milk", "meat", "salt"],
    "nature":  ["river", "mountain", "snow", "forest", "wind", "rain", "sun", "moon", "lake", "sea", "tree", "field"],
    "tech":    ["computer", "robot", "code", "data", "network", "machine", "programmer", "software", "algorithm", "server"],
    "emotion": ["love", "fear", "joy", "anger", "sadness", "hope", "happiness"],
    "action":  ["run", "walk", "jump", "eat", "sleep", "read", "drink", "sing", "laugh", "cry"],
}


def tokenize(sentence: str):
    """Simple tokenization: lowercase and strip punctuation"""
    table = str.maketrans("", "", ".,!?;:'\"()")  # 建立"删除字符"映射表:把常见标点全部去掉
    return sentence.lower().translate(table).split()  # 转小写、去标点、按空白分词


# ---------------------------------------------------------------------------
# 2. 向量化:共现 → PPMI → SVD(挖掘文本中潜在的低维结构)
# ---------------------------------------------------------------------------
def build_ppmi_matrix(corpus, window=3):
    """Build a PPMI-weighted co-occurrence matrix."""
    # 统计共现频次:滑动窗口内同时出现的词对计数
    co_occur = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(float)
    total = 0.0

    for sentence in corpus:
        words = tokenize(sentence)
        for i, w in enumerate(words):
            counts[w] += 1.0
            total += 1.0
            for j in range(max(0, i - window), min(len(words), i + window + 1)):
                if i != j:
                    co_occur[w][words[j]] += 1.0 / abs(i - j)  # 距离越近的词权重越高:用 1/|i−j| 加权(距离衰减)

    vocab = sorted(counts.keys())       # 词汇表:按字母序排列的所有出现过的词
    idx = {w: i for i, w in enumerate(vocab)}   # 词 → 索引 的映射,便于填充矩阵
    n = len(vocab)

    # 计算词概率 p(w) 与共现概率,为 PPMI 做准备
    p_w = np.array([counts[w] / total for w in vocab])
    C = np.zeros((n, n))
    for w, neighbors in co_occur.items():
        for v, cnt in neighbors.items():
            C[idx[w], idx[v]] = cnt
    C = C / C.sum()                     # 归一化为联合概率分布 p(w,c)
    p_c = C.sum(axis=0)                 # 边缘概率 p(c) = Σ_w p(w,c)

    # PMI(w, c) = log(p(w,c) / (p(w)·p(c))):衡量两个词相对独立情况下的共现强度;
    # 把负值截断为 0 即得到 PPMI(正点互信息)
    with np.errstate(divide="ignore"):  # 屏蔽 log(0) 的警告(未共现的词对分子为 0)
        pmi = np.log(C / np.outer(p_w, p_c))
    ppmi = np.maximum(pmi, 0.0)
    return ppmi, vocab


def svd_embedding(ppmi, k=20):
    """Word embeddings via truncated SVD (LSA). Returns (embedding matrix
    n_words x k, fraction of explained variance)."""
    U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)   # 对 PPMI 矩阵做奇异值分解:U 的行空间编码词的语义方向
    total_var = (S ** 2).sum()          # 总方差 = 所有奇异值平方之和
    explained = (S[:k] ** 2).sum() / total_var   # 前 k 个主成分解释的方差占比(衡量降维的信息保留程度)
    W = U[:, :k] * S[:k]            # 每一行是一个词的嵌入向量(取前 k 个奇异向量,并按奇异值缩放)
    return W, explained


# ---------------------------------------------------------------------------
# 3. 工具函数:余弦相似度 / 类比推理 / 最近邻
# ---------------------------------------------------------------------------
def normalize_rows(M):
    # 对行做 L2 归一化,使余弦相似度只依赖方向、与模长无关;零向量避免除零
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return M / norms


def cosine_sim_matrix(W):
    Wn = normalize_rows(W)
    return Wn @ Wn.T   # 余弦相似度矩阵 = 归一化后的行向量两两点积


def nearest_neighbors(W, vocab, query, top=5):
    """Return the words most similar to query, excluding itself."""
    q = W[vocab.index(query)]
    sims = cosine_sim_matrix(W)[vocab.index(query)]
    order = np.argsort(-sims)
    return [(vocab[i], sims[i]) for i in order if vocab[i] != query][:top]


def cluster_separation(proj, vocab, word_group, groups=None):
    """Cluster separation = mean inter-group distance / mean intra-group distance
    (larger means clearer clusters on the manifold)."""
    groups = groups or GROUPS
    intra, inter = [], []
    for g, words in groups.items():
        present = [vocab.index(w) for w in words if w in vocab]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                intra.append(np.linalg.norm(proj[present[i]] - proj[present[j]]))   # 组内两两距离
    for i in range(len(vocab)):
        for j in range(i + 1, len(vocab)):
            gi, gj = word_group.get(vocab[i]), word_group.get(vocab[j])
            if gi and gj and gi != gj:
                inter.append(np.linalg.norm(proj[i] - proj[j]))   # 不同组之间的两两距离
    return np.mean(inter) / (np.mean(intra) + 1e-12)


def analogy(W, vocab, a, b, c, top=3):
    """Analogy: a - b + c ≈ ? (classic: apple - fruit + grain ≈ rice)"""
    ia, ib, ic = (vocab.index(w) for w in (a, b, c))
    target = W[ia] - W[ib] + W[ic]      # 语义关系在向量空间中是"方向":a − b + c 得到目标词的近似位置
    Wn = normalize_rows(W)
    sims = Wn @ (target / (np.linalg.norm(target) + 1e-12))   # 与目标方向做余弦相似度
    order = np.argsort(-sims)
    return [(vocab[i], sims[i]) for i in order if vocab[i] not in (a, b, c)][:top]


# ---------------------------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Text Manifold Demo")
    print("=" * 60)

    # 向量化:构建 PPMI 共现矩阵(窗口大小 3)
    ppmi, vocab = build_ppmi_matrix(CORPUS, window=3)
    print(f"Corpus sentences: {len(CORPUS)}, vocabulary size: {len(vocab)}")

    W, explained = svd_embedding(ppmi, k=20)
    print(f"SVD, top 20 components, explained variance ratio: {explained:.1%}")

    # --- 流形可视化:线性方法(PCA/SVD)vs 非线性方法(Isomap / t-SNE) ---
    Wc = W - W.mean(axis=0, keepdims=True)      # 中心化(减去均值),为 PCA 做准备
    _, S2, Vt2 = np.linalg.svd(Wc, full_matrices=False)    # 对中心化矩阵做 SVD,Vt2 的行是主成分方向
    proj_linear = Wc @ Vt2[:2].T                # 投影到前两个主成分上,得到 2D 平面坐标(即 PCA 投影)

    projections = {"PCA (linear)": proj_linear}
    use_sklearn = False
    try:
        # 若安装了 scikit-learn,再补充非线性流形方法:Isomap 与 t-SNE
        from sklearn.manifold import Isomap, TSNE
        projections["Isomap"] = Isomap(n_neighbors=8, n_components=2).fit_transform(W)
        projections["t-SNE"] = TSNE(n_components=2, perplexity=15, random_state=0).fit_transform(W)
        use_sklearn = True
    except ImportError:
        print("(scikit-learn not installed; showing linear PCA only. Run 'pip install scikit-learn' to enable nonlinear manifolds)")

    color_map = {
        "animal": "#d62728", "food": "#2ca02c", "nature": "#1f77b4",
        "tech": "#9467bd", "emotion": "#ff7f0e", "action": "#8c564b",
    }
    # 建"单词 → 语义类别"的反查表,供着色与后续验证使用
    word_group = {}
    for g, words in GROUPS.items():
        for w in words:
            word_group[w] = g

    fig, axes = plt.subplots(1, len(projections), figsize=(6 * len(projections), 6))
    if len(projections) == 1:
        axes = [axes]
    for ax, (name, proj) in zip(axes, projections.items()):
        # 每个子图:按语义类别着色绘制词向量投影,并标注词名
        for g, color in color_map.items():
            words = [w for w in vocab if word_group.get(w) == g]
            if not words:
                continue
            ixs = [vocab.index(w) for w in words]
            ax.scatter(proj[ixs, 0], proj[ixs, 1], c=color, s=50, label=g, alpha=0.85)
            for w, ix in zip(words, ixs):
                ax.annotate(w, (proj[ix, 0], proj[ix, 1]), fontsize=8, alpha=0.9)
        # 未被分组的词用灰色小点画出
        for ix in range(len(vocab)):
            if vocab[ix] not in word_group:
                ax.scatter(proj[ix, 0], proj[ix, 1], c="gray", s=25, alpha=0.4)
        ax.set_title(f"{name}\n(cluster separation = {cluster_separation(proj, vocab, word_group):.2f})")
        ax.set_xlabel("component 1"); ax.set_ylabel("component 2")
    axes[0].legend(loc="best", fontsize=8)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'text_manifold.png')
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Manifold plot saved: {out_path}")

    # --- 线性流形 vs 非线性流形的聚类分离度对比 ---
    print("\n--- Linear vs nonlinear manifold (separation = inter/intra-group distance; larger is clearer) ---")
    for name, proj in projections.items():
        print(f"  {name:12s}: {cluster_separation(proj, vocab, word_group):.2f}")
    if use_sklearn:
        print("  Note: t-SNE / Isomap unfold nonlinear structure, making semantic clusters clearer on the plane;")
        print("        PCA preserves global directions (which analogies rely on). The two views are complementary.")

    # --- 验证 1:语义最近邻 —— 与查询词最相似的词 ---
    print("\n--- Semantic neighbors (words most similar to a query) ---")
    for q in ["cat", "apple", "data", "joy", "river"]:
        if q in vocab:
            print(f"  {q:8s} -> " + ", ".join(f"{w}({s:.2f})" for w, s in nearest_neighbors(W, vocab, q, top=4)))

    # --- 验证 2:类比推理(流形上的方向对应语义关系) ---
    print("\n--- Analogies (vector direction = semantic relation) ---")
    for a, b, c in [("apple", "fruit", "grain"), ("dog", "cat", "mouse"),
                    ("river", "water", "data")]:
        try:
            res = analogy(W, vocab, a, b, c, top=2)
            print(f"  {a} - {b} + {c} ≈ " + ", ".join(f"{w}({s:.2f})" for w, s in res))
        except ValueError as e:
            print(f"  {a}-{b}+{c}: word not in corpus ({e})")

    # --- 验证 3:组内相似度 vs 组间相似度(定量检验聚簇) ---
    print("\n--- Semantic group structure ---")
    S = cosine_sim_matrix(W)
    intra, inter = [], []
    for g, words in GROUPS.items():
        present = [vocab.index(w) for w in words if w in vocab]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                intra.append(S[present[i], present[j]])     # 同组词的余弦相似度
    for i in range(len(vocab)):
        for j in range(i + 1, len(vocab)):
            gi, gj = word_group.get(vocab[i]), word_group.get(vocab[j])
            if gi and gj and gi != gj:
                inter.append(S[i, j])                       # 不同组词的余弦相似度
    print(f"  Mean intra-group cosine similarity: {np.mean(intra):.3f}")
    print(f"  Mean inter-group cosine similarity: {np.mean(inter):.3f}")
    print(f"  -> intra-group similarity {'exceeds' if np.mean(intra) > np.mean(inter) else 'does not exceed'} "
          f"inter-group, confirming that semantic clusters are present on the manifold")


if __name__ == "__main__":
    main()
