import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


from datafun.lego import get_X_theme_for_set_names, load_lego_data, select_examples

st.title('Entscheidungsbäume')

st.page_link('pages/data_lego.py', label='Datenquelle: Lego Database')

D = load_lego_data()
df_themes, df_sets, df_inventories, df_inv_parts, df_color_list, df_parts, df_part_types, themes_root, color_list, part_types_ids, part_types_names = D

themes_A, themes_B, min_parts = select_examples(D)

X_A, X_list_A, set_names_A, set_nums_A = get_X_theme_for_set_names(D, themes_A, min_parts=min_parts)
X_B, X_list_B, set_names_B, set_nums_B = get_X_theme_for_set_names(D, themes_B, min_parts=min_parts)

n_a = X_A.shape[0]
n_b = X_B.shape[0]
n = n_a + n_b
X = np.vstack((X_A, X_B))
d = X.shape[1]

y = np.zeros(n)
y[n_a:] = 1

set_names = np.array(list(set_names_A) + list(set_names_B), dtype=object)



st.write(r'''
    $n$: Anzahl Datensätze
         
    $k$: Anzahl Klassen
         
    $d$: Anzahl Dimensionen

    Sei $(x, y)$ ein Element des Datensatzes $D$.
    So ist die Entropie $H$ mit den Klassen $Z = {c_1, ..., c_k}$ definiert durch

    $H(D) = - \sum_{i=1}^k \frac{\overbrace{|\{(x, y)\in D \text{ mit } y=c_i \}|}^{\substack{\text{Anzahl Datensätze} \\ \text{der Klasse } c_i}}}{\underbrace{|D|}_{\text{Anzahl aller Datensätze}}} \log \underbrace{\frac{|\{(x, y)\in D \text{ mit } y=c_i \}|}{|D|}}_{\text{dasselbe aber mit Logarithmus}}$
''')

def entropy(y, log=np.log2):

    Z = np.unique(y) # all unique classes in y
    H = 0.0
    for c_i in Z:
        p_c = np.sum(y == c_i) / y.size
        if p_c > 0: # avoid log(0) because lim x -> 0: x * log(x) = 0
            H += -p_c * log(p_c)

    return H

st.write(r'''
    $r$: Anzahl unterschiedlicher Merkmalausprägungen

    Seien $\tilde{x}_{i,1}, ..., \tilde{x}_{i,r}$ alle unterschiedlichen Merkmalausprägungen einer Dimension $i$ von $x$
    so ist die bedinge Entropie $H(D | i)$ definiert durch

    $H(D | i) = \sum_{j=1}^r \frac{\overbrace{|\{(x_1, ..., x_n, y)\in D \text{ mit } x_i=\tilde{x}_{i,j} \}|}^{\substack{\text{Anzahl Datensätze} \\ \text{der Ausprägung } \tilde{x}_{i,j}}}}{\underbrace{|D|}_{\text{Anzahl aller Datensätze}}} H(\underbrace{\{(x_1, ..., x_n, y)\in D \text{ mit } x_i=\tilde{x}_{i,j} \}}_{\substack{\text{Nur die Datensätze} \\ \text{mit Ausprägung }  \tilde{x}_{i,j}}})$

    Der Informationsgewinn $IG(D, i)$ einer Dimension $i$ ist dann als die Differenz zur Entropie $H(D)$ definiert

    $IG(D, i) = H(D) - H(D | i)$
''')

def conditional_entropy(x, y, log=np.log2):

    x_unique = np.unique(x)

    H = 0.0
    for x_j in x_unique:
        x_j_idx = np.argwhere(x == x_j).flatten()
        H += x_j_idx.size / x.size * entropy(y[x_j_idx], log)

    return H

def information_gain(x, y, log=np.log2):
    return entropy(y, log) - conditional_entropy(x, y, log)


if len(set_names) == 0:
    st.warning('Es wurden kein Set ausgewählt. Bitte wähle mindestens ein Lego-Set aus, um fortzufahren.')
    st.stop()

# Shuffle samples jointly and split into two equal halves.
random_seed = st.slider('Random Seed', min_value=0, max_value=100, value=42, step=1)
rng = np.random.default_rng(random_seed)
perm = rng.permutation(n)
X_shuffled = X[perm]
y_shuffled = y[perm]
set_names_shuffled = set_names[perm]

n_half = X_shuffled.shape[0] // 2
X_train, X_test = X_shuffled[:n_half], X_shuffled[n_half:]
y_train, y_test = y_shuffled[:n_half], y_shuffled[n_half:]
set_names_train, set_names_test = set_names_shuffled[:n_half], set_names_shuffled[n_half:]

n_test = y_test.size

@dataclass
class DecisionTreeNode:
    X: np.ndarray
    y: np.ndarray
    set_names: np.ndarray
    depth: int

    best_IG: float = 0.0
    feature_idx: int | None = None
    feature_name: str | None = None
    split_value: float = 0.0

    left: "DecisionTreeNode | None" = None
    right: "DecisionTreeNode | None" = None

def information_gain_split(X, y, set_names, depth=0, max_depth=5, subsampling=None):
    node = DecisionTreeNode(
        X=X,
        y=y,
        set_names=set_names,
        depth=depth
    )

    d = X.shape[1]
    IG_j_max = 0.0
    IG_j_max_idx = -1
    IG_j_max_split_value = 0.0
    perm = rng.permutation(d)
    d_sub = d
    if subsampling == 'sqrt':
        d_sub = int(np.sqrt(d))
    for j in perm[:d_sub]:
    # if True:
    #     j = 10
        x_j = X[:,j]


        # go only further if there are at least 2 different values in the dimension
        if np.unique(x_j).size == 1:
            continue

        nonzero_idx = np.argwhere(x_j > 0).flatten()
        IG_i_max = 0.0
        IG_i_max_split_value = 0.0
        for i in nonzero_idx:

            split_value = x_j[i]
            lower_idx = np.argwhere(x_j <= split_value)
            upper_idx = np.argwhere(x_j > split_value)

            H_lower = lower_idx.size / x_j.size * entropy(y[lower_idx])
            H_upper = upper_idx.size / x_j.size * entropy(y[upper_idx])
            # st.write(lower_idx, upper_idx)

            # if lower_idx.size == 0 or upper_idx.size == 0:
            #     continue

            H = entropy(y)

            IG = H - H_lower - H_upper
            if IG > IG_i_max:
                IG_i_max = IG
                IG_i_max_split_value = split_value

        if IG_i_max > IG_j_max:
            IG_j_max = IG_i_max
            IG_j_max_idx = j
            IG_j_max_split_value = IG_i_max_split_value

        # st.write(f'{j}: {part_types_names[j]} - size: {x_j.size} / lower: {lower_idx.size} / upper: {upper_idx.size} - IG: {IG_i_max:.4f}')

    # st.write(f'**{depth}**: Bestes Merkmal: {part_types_names[IG_j_max_idx]} mit IG: {IG_j_max:.4f} und Split Value: {IG_j_max_split_value}')

    node.best_IG = IG_j_max
    node.feature_idx = IG_j_max_idx
    node.feature_name = part_types_names[IG_j_max_idx]
    node.split_value = IG_j_max_split_value

    if IG_j_max == 0.0 or depth >= max_depth:
        # st.write(np.unique(y, return_counts=True))
        return node

    lower_idx = np.argwhere(X[:, IG_j_max_idx] <= IG_j_max_split_value).flatten()
    upper_idx = np.argwhere(X[:, IG_j_max_idx] > IG_j_max_split_value).flatten()

    # st.write(f'{lower_idx.size} / {upper_idx.size}')

    X_lower = X[lower_idx,:]
    set_names_lower = set_names[lower_idx]
    X_upper = X[upper_idx,:]
    set_names_upper = set_names[upper_idx]

    # st.write(set_names_lower)
    # st.write(set_names_upper)

    # st.write(f'Anzahl Datensätze in der unteren Gruppe: {lower_idx.size}')
    # st.write(f'Anzahl Datensätze in der oberen Gruppe: {upper_idx.size}')

    # lower_classes, lower_counts = np.unique(y[lower_idx], return_counts=True)
    # upper_classes, upper_counts = np.unique(y[upper_idx], return_counts=True)
    # st.write(f'Klassenverteilung in der unteren Gruppe: {dict(zip(lower_classes, lower_counts))}')
    # st.write(f'Klassenverteilung in der oberen Gruppe: {dict(zip(upper_classes, upper_counts))}')

    node.left = information_gain_split(X_lower, y[lower_idx], set_names_lower, depth + 1, max_depth=max_depth, subsampling=subsampling)
    node.right = information_gain_split(X_upper, y[upper_idx], set_names_upper, depth + 1, max_depth=max_depth, subsampling=subsampling)

    return node

def random_forest(_X, _y, _set_names, n_trees=10, max_depth=5, random_seed=42):
    rng = np.random.default_rng(random_seed)
    forest = []

    if n_trees == 1:
        tree = information_gain_split(_X, _y, _set_names, max_depth=max_depth, subsampling=None)
        forest.append(tree)
        return forest

    progress = st.progress(0)
    for i in range(n_trees):
        progress.progress((i + 1) / n_trees)
        # Bootstrap sample
        n_samples = _X.shape[0]
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_sample = _X[indices]
        y_sample = _y[indices]
        set_names_sample = _set_names[indices]

        tree = information_gain_split(X_sample, y_sample, set_names_sample, max_depth=max_depth, subsampling='sqrt')
        forest.append(tree)

    progress.empty()

    return forest


n_trees = st.slider('Amount of trees in the forest', min_value=1, max_value=100, value=10, step=1)

forest = random_forest(X_train, y_train, set_names_train, n_trees=n_trees, max_depth=5, random_seed=random_seed)
# decision_tree = information_gain_split(X_train, y_train, set_names_train)

def predict_by_tree(x_i, decision_tree: DecisionTreeNode):
    node = decision_tree
    y_pred_i = None

    while node is not None:

        if node.left is None and node.right is None:
            classes, counts = np.unique(node.y, return_counts=True)
            y_pred_i = classes[np.argmax(counts)]
            break

        if x_i[node.feature_idx] <= node.split_value:
            node = node.left
        else:
            node = node.right

    return y_pred_i

def predict_by_forest(x_i, forest):
    votes = []
    for tree in forest:
        y_pred_i = predict_by_tree(x_i, tree)
        votes.append(y_pred_i)

    # st.write(np.bincount(votes))

    return np.bincount(votes).argmax()

# st.write(y_test, y_pred)



## Plotting the decision tree

def _count_leaves(node: DecisionTreeNode) -> int:
    if node.left is None and node.right is None:
        return 1
    n_left = _count_leaves(node.left) if node.left is not None else 0
    n_right = _count_leaves(node.right) if node.right is not None else 0
    return n_left + n_right


def _max_depth(node: DecisionTreeNode) -> int:
    left_depth = _max_depth(node.left) if node.left is not None else node.depth
    right_depth = _max_depth(node.right) if node.right is not None else node.depth
    return max(node.depth, left_depth, right_depth)


def _assign_positions(node: DecisionTreeNode, x_cursor: list[float], pos: dict[int, tuple[float, float]]):
    if node.left is None and node.right is None:
        x = x_cursor[0]
        pos[id(node)] = (x, -node.depth)
        x_cursor[0] += 1.0
        return

    if node.left is not None:
        _assign_positions(node.left, x_cursor, pos)
    if node.right is not None:
        _assign_positions(node.right, x_cursor, pos)

    child_x = []
    if node.left is not None:
        child_x.append(pos[id(node.left)][0])
    if node.right is not None:
        child_x.append(pos[id(node.right)][0])
    pos[id(node)] = (sum(child_x) / len(child_x), -node.depth)


def plot_decision_tree(root: DecisionTreeNode):
    n_leaves = _count_leaves(root)
    tree_depth = _max_depth(root)

    fig_w = max(10, 2.2 * n_leaves)
    fig_h = max(6, 2.0 * (tree_depth + 1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    pos = {}
    _assign_positions(root, [0.0], pos)

    def _node_text(node: DecisionTreeNode) -> str:
        if node.left is None and node.right is None:
            # st.write(node.set_names)
            classes, counts = np.unique(node.y, return_counts=True)
            class_names = [", ".join(themes_A), ", ".join(themes_B)]
            dist = "\n".join([f"{class_names[int(c)]}:{(n/node.y.size)*100:.2f}%" for c, n in zip(classes, counts)])
            set_names_str = "\n".join(node.set_names)
            if len(node.set_names) > 5:
                set_names_str = "\n".join(node.set_names[:5]) + "\n..."
            return f"{dist}\nn={node.y.size}\n{set_names_str}"

        feature = node.feature_name if node.feature_name is not None else "-"
        # split = node.split_value if node.split_value is not None else 0.0
        return f"{feature}\nIG={node.best_IG:.3f}\nn={node.y.size}"

    def _draw(node: DecisionTreeNode):
        x, y_pos = pos[id(node)]

        # for child, edge_label in ((node.left, f"<= {node.split_value*100:.2f}%"), (node.right, f"> {node.split_value*100:.2f}%")):
        for child, edge_label in ((node.left, f"<= {node.split_value:.0f}"), (node.right, f"> {node.split_value:.0f}")):
            if child is None:
                continue
            cx, cy = pos[id(child)]
            ax.plot([x, cx], [y_pos, cy], color="#4a5568", linewidth=1.8, zorder=1)
            mx, my = (x + cx) / 2.0, (y_pos + cy) / 2.0
            ax.text(
                mx,
                my + 0.08,
                edge_label,
                fontsize=9,
                color="#2d3748",
                ha="center",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",   # oder z.B. "#cbd5e1" für einen Rand
                    boxstyle="round,pad=0.15",
                    alpha=1.0
                )
            )
            _draw(child)

        is_leaf = node.left is None and node.right is None
        box_fc = "#fef3c7" if is_leaf else "#dbeafe"
        ax.text(
            x,
            y_pos,
            _node_text(node),
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=box_fc, edgecolor="#1f2937", linewidth=1.2),
            zorder=2,
        )

    _draw(root)
    ax.set_xlim(-0.7, max(0.7, n_leaves - 0.3))
    ax.set_ylim(-(tree_depth + 0.8), 0.8)
    ax.set_title("Entscheidungsbaum (Feature-Splits)", fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    return fig


plot_tree_idx = st.slider('Index of the tree to plot', min_value=1, max_value=n_trees, value=n_trees//2, step=1)

st.pyplot(plot_decision_tree(forest[plot_tree_idx - 1]), clear_figure=True)




y_pred = np.zeros(n_test)
for i in range(n_test):
    y_pred[i] = predict_by_forest(X_test[i,:], forest)


# Step 1: indices where y_test and y_pred are both 1 (true positives).
true_positive = np.argwhere((y_test == 1) & (y_pred == 1)).flatten()
true_negative = np.argwhere((y_test == 0) & (y_pred == 0)).flatten()
false_positive = np.argwhere((y_test == 0) & (y_pred == 1)).flatten()
false_negative = np.argwhere((y_test == 1) & (y_pred == 0)).flatten()


n_pos_test = sum(y_test == 1)
n_neg_test = sum(y_test == 0)

confusion_matrix = np.array([[len(true_positive) / n_pos_test,  len(false_positive) / n_neg_test], 
                             [len(false_negative) / n_pos_test, len(true_negative) / n_neg_test]])
confusion_labels = np.array([['True Positive', 'False Positive'], 
                             ['False Negative', 'True Negative']])


fig, ax = plt.subplots()
im = ax.imshow(confusion_matrix)
ax.set_xticks(np.arange(2))
ax.set_yticks(np.arange(2))
ax.set_xticklabels([r'$y=1$', r'$y=0$'])
ax.set_yticklabels([r'$\hat{y} > 0.5$', r'$\hat{y} < 0.5$'])
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, f'{confusion_labels[i, j]}\n{confusion_matrix[i, j]*100:.2f} %', ha='center', va='center', color='black')

ax.set_title('Confusion Matrix')
st.pyplot(fig)