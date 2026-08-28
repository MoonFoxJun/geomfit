"""文字蕴含的流形 Demo —— 从语料构建语义空间并可视化其低维结构

思想（分布假说）：一个词的意义 = 它周围经常一起出现的词。
流程（这是"别人已经做到的"标准做法，即 LSA / 潜在语义分析）：
    语料 → 分词 → 共现矩阵 → PPMI 加权 → SVD 降维 → 词向量
    词向量张成的低维空间就是"文字的流形"，其中：
      - 语义相近的词聚在一起（流形上的簇）
      - 语义关系表现为向量方向（类比推理：apple - fruit + grain ≈ rice）

与触摸数据的联系：触摸(压力/面积/时间序列) → 特征向量 → 同样这套
降维/流形方法，输入换成传感器数据即可，流水线完全一致。
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
# 1. 语料准备：手写小语料，包含几个明显的语义群（动物/食物/自然/科技/情绪/动作）
# ---------------------------------------------------------------------------
CORPUS = [
    # 动物 & 自然
    "the cat and the dog play in the garden",
    "the mouse hides from the cat",
    "a lion lives in the jungle",
    "the tiger hunts in the forest",
    "the bear sleeps in the cave",
    "the wolf runs with the pack",
    "the horse and the cow live on the farm",
    "birds sing in the morning",
    "the dog runs after the cat",
    # 食物
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
    # 自然
    "the river flows to the sea",
    "the mountain is covered with snow",
    "the forest is full of trees",
    "the wind blows across the field",
    "the rain falls on the roof",
    "the sun rises in the east",
    "the moon shines at night",
    "the lake reflects the sky",
    # 科技
    "the computer runs the code",
    "the robot moves like a machine",
    "data flows through the network",
    "the programmer writes software",
    "the algorithm processes the data",
    "the machine learns from data",
    "the server stores the files",
    # 情绪
    "love gives us joy and hope",
    "fear makes the heart beat fast",
    "anger turns into sadness",
    "we feel happiness when we see the sun",
    "the news brings us joy",
    "she cried with sadness",
    # 动作
    "we run and walk in the park",
    "the child jumps and laughs",
    "i read a book before sleep",
    "he eats dinner and drinks water",
    "we sleep under the stars",
    "she reads the letter with hope",
]

# 语义群（用于着色与验证）
GROUPS = {
    "animal":  ["cat", "dog", "lion", "tiger", "bear", "wolf", "mouse", "horse", "cow", "bird"],
    "food":    ["bread", "cheese", "apple", "banana", "rice", "wheat", "grain", "fruit", "milk", "meat", "salt"],
    "nature":  ["river", "mountain", "snow", "forest", "wind", "rain", "sun", "moon", "lake", "sea", "tree", "field"],
    "tech":    ["computer", "robot", "code", "data", "network", "machine", "programmer", "software", "algorithm", "server"],
    "emotion": ["love", "fear", "joy", "anger", "sadness", "hope", "happiness"],
    "action":  ["run", "walk", "jump", "eat", "sleep", "read", "drink", "sing", "laugh", "cry"],
}


def tokenize(sentence: str):
    """简单分词：小写 + 去标点"""
    table = str.maketrans("", "", ".,!?;:'\"()")
    return sentence.lower().translate(table).split()


# ---------------------------------------------------------------------------
# 2. 向量化：共现矩阵 → PPMI → SVD（这就是"文字蕴含的低维结构"）
# ---------------------------------------------------------------------------
def build_ppmi_matrix(corpus, window=3):
    """构建 PPMI 加权共现矩阵。"""
    # 共现统计
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
                    co_occur[w][words[j]] += 1.0 / abs(i - j)  # 距离越近权重越高

    vocab = sorted(counts.keys())
    idx = {w: i for i, w in enumerate(vocab)}
    n = len(vocab)

    # 概率
    p_w = np.array([counts[w] / total for w in vocab])
    C = np.zeros((n, n))
    for w, neighbors in co_occur.items():
        for v, cnt in neighbors.items():
            C[idx[w], idx[v]] = cnt
    C = C / C.sum()
    p_c = C.sum(axis=0)

    # PMI(w, c) = log(p(w,c) / (p(w) p(c)))，负值截断为 0 → PPMI
    with np.errstate(divide="ignore"):
        pmi = np.log(C / np.outer(p_w, p_c))
    ppmi = np.maximum(pmi, 0.0)
    return ppmi, vocab


def svd_embedding(ppmi, k=20):
    """SVD 降维得到词向量（LSA）。返回 (词向量矩阵 n_words x k, 解释方差比例)。"""
    U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
    total_var = (S ** 2).sum()
    explained = (S[:k] ** 2).sum() / total_var
    W = U[:, :k] * S[:k]            # 行向量 = 词的嵌入
    return W, explained


# ---------------------------------------------------------------------------
# 3. 工具：余弦相似度 / 类比 / 近邻
# ---------------------------------------------------------------------------
def normalize_rows(M):
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return M / norms


def cosine_sim_matrix(W):
    Wn = normalize_rows(W)
    return Wn @ Wn.T


def nearest_neighbors(W, vocab, query, top=5):
    """返回与 query 最相似的词（不含自身）。"""
    q = W[vocab.index(query)]
    sims = cosine_sim_matrix(W)[vocab.index(query)]
    order = np.argsort(-sims)
    return [(vocab[i], sims[i]) for i in order if vocab[i] != query][:top]


def cluster_separation(proj, vocab, word_group, groups=None):
    """聚类分离度 = 跨群平均距离 / 同群平均距离（越大说明流形上的簇越清晰）。"""
    groups = groups or GROUPS
    intra, inter = [], []
    for g, words in groups.items():
        present = [vocab.index(w) for w in words if w in vocab]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                intra.append(np.linalg.norm(proj[present[i]] - proj[present[j]]))
    for i in range(len(vocab)):
        for j in range(i + 1, len(vocab)):
            gi, gj = word_group.get(vocab[i]), word_group.get(vocab[j])
            if gi and gj and gi != gj:
                inter.append(np.linalg.norm(proj[i] - proj[j]))
    return np.mean(inter) / (np.mean(intra) + 1e-12)


def analogy(W, vocab, a, b, c, top=3):
    """类比：a - b + c ≈ ?（经典：apple - fruit + grain ≈ rice）"""
    ia, ib, ic = (vocab.index(w) for w in (a, b, c))
    target = W[ia] - W[ib] + W[ic]
    Wn = normalize_rows(W)
    sims = Wn @ (target / (np.linalg.norm(target) + 1e-12))
    order = np.argsort(-sims)
    return [(vocab[i], sims[i]) for i in order if vocab[i] not in (a, b, c)][:top]


# ---------------------------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("文字蕴含的流形 Demo")
    print("=" * 60)

    # 向量化
    ppmi, vocab = build_ppmi_matrix(CORPUS, window=3)
    print(f"语料句子数: {len(CORPUS)}，词汇量: {len(vocab)}")

    W, explained = svd_embedding(ppmi, k=20)
    print(f"SVD 取前 20 维，解释方差比例: {explained:.1%}")

    # --- 流形可视化：线性 (PCA/SVD) vs 非线性 (Isomap / t-SNE) ---
    Wc = W - W.mean(axis=0, keepdims=True)
    _, S2, Vt2 = np.linalg.svd(Wc, full_matrices=False)
    proj_linear = Wc @ Vt2[:2].T

    projections = {"PCA (linear)": proj_linear}
    use_sklearn = False
    try:
        from sklearn.manifold import Isomap, TSNE
        projections["Isomap"] = Isomap(n_neighbors=8, n_components=2).fit_transform(W)
        projections["t-SNE"] = TSNE(n_components=2, perplexity=15, random_state=0).fit_transform(W)
        use_sklearn = True
    except ImportError:
        print("(未安装 scikit-learn，仅显示线性 PCA，可运行 pip install scikit-learn 解锁非线性流形)")

    color_map = {
        "animal": "#d62728", "food": "#2ca02c", "nature": "#1f77b4",
        "tech": "#9467bd", "emotion": "#ff7f0e", "action": "#8c564b",
    }
    word_group = {}
    for g, words in GROUPS.items():
        for w in words:
            word_group[w] = g

    fig, axes = plt.subplots(1, len(projections), figsize=(6 * len(projections), 6))
    if len(projections) == 1:
        axes = [axes]
    for ax, (name, proj) in zip(axes, projections.items()):
        for g, color in color_map.items():
            words = [w for w in vocab if word_group.get(w) == g]
            if not words:
                continue
            ixs = [vocab.index(w) for w in words]
            ax.scatter(proj[ixs, 0], proj[ixs, 1], c=color, s=50, label=g, alpha=0.85)
            for w, ix in zip(words, ixs):
                ax.annotate(w, (proj[ix, 0], proj[ix, 1]), fontsize=8, alpha=0.9)
        # 未分组的词用灰色
        for ix in range(len(vocab)):
            if vocab[ix] not in word_group:
                ax.scatter(proj[ix, 0], proj[ix, 1], c="gray", s=25, alpha=0.4)
        ax.set_title(f"{name}\n(cluster separation = {cluster_separation(proj, vocab, word_group):.2f})")
        ax.set_xlabel("component 1"); ax.set_ylabel("component 2")
    axes[0].legend(loc="best", fontsize=8)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'text_manifold.png')
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"流形图已保存: {out_path}")

    # --- 线性 vs 非线性流形对比 ---
    print("\n--- 线性 vs 非线性流形对比（分离度 = 跨群距离/同群距离，越大越清晰） ---")
    for name, proj in projections.items():
        print(f"  {name:12s}: {cluster_separation(proj, vocab, word_group):.2f}")
    if use_sklearn:
        print("  说明: t-SNE / Isomap 能展开非线性结构，让语义簇在平面上更清晰；")
        print("        PCA 保留全局方向（类比推理依赖它）。两者视角互补。")

    # --- 结构验证 1：语义近邻 ---
    print("\n--- 语义近邻（与某词最相似的词） ---")
    for q in ["cat", "apple", "data", "joy", "river"]:
        if q in vocab:
            print(f"  {q:8s} -> " + ", ".join(f"{w}({s:.2f})" for w, s in nearest_neighbors(W, vocab, q, top=4)))

    # --- 结构验证 2：类比推理（流形上的"方向"） ---
    print("\n--- 类比推理（向量方向 = 语义关系） ---")
    for a, b, c in [("apple", "fruit", "grain"), ("dog", "cat", "mouse"),
                    ("river", "water", "data")]:
        try:
            res = analogy(W, vocab, a, b, c, top=2)
            print(f"  {a} - {b} + {c} ≈ " + ", ".join(f"{w}({s:.2f})" for w, s in res))
        except ValueError as e:
            print(f"  {a}-{b}+{c}: 词汇不在语料中 ({e})")

    # --- 结构验证 3：群内 vs 群间相似度 ---
    print("\n--- 语义群结构验证 ---")
    S = cosine_sim_matrix(W)
    intra, inter = [], []
    for g, words in GROUPS.items():
        present = [vocab.index(w) for w in words if w in vocab]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                intra.append(S[present[i], present[j]])
    for i in range(len(vocab)):
        for j in range(i + 1, len(vocab)):
            gi, gj = word_group.get(vocab[i]), word_group.get(vocab[j])
            if gi and gj and gi != gj:
                inter.append(S[i, j])
    print(f"  同群平均余弦相似度: {np.mean(intra):.3f}")
    print(f"  跨群平均余弦相似度: {np.mean(inter):.3f}")
    print(f"  -> 同群相似度 {'显著高于' if np.mean(intra) > np.mean(inter) else '未高于'}跨群，说明流形上确实存在语义簇")


if __name__ == "__main__":
    main()
