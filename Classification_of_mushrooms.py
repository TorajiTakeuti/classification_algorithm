import numpy as np
import sys
# 1. 情報エントロピーの計算
def calculate_entropy(y):
    if len(y) == 0:
        return 0
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

def ask_question(prompt):
    """質問をして 1(yes) か 0(no) を返す。それ以外は終了。"""
    answer = input(prompt).lower() # 小文字に統一して判定
    if answer == "yes":
        return 1
    elif answer == "no":
        return 0
    else:
        print("質問に答えろ（プログラムを終了します）")
        sys.exit()

# 2. 決定木アルゴリズム
class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """学習（行列Xとラベルyを使用して木を構築）"""
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        """【再帰】学習プロセス"""
        num_samples, num_features = X.shape
        if len(np.unique(y)) == 1 or depth >= self.max_depth:
            return np.bincount(y).argmax()

        best_gain = -1
        best_split = None
        current_entropy = calculate_entropy(y)

        for feature_idx in range(num_features):
            values = np.unique(X[:, feature_idx])
            for threshold in values:
                left_indices = np.where(X[:, feature_idx] <= threshold)[0]
                right_indices = np.where(X[:, feature_idx] > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                e_left = calculate_entropy(y[left_indices])
                e_right = calculate_entropy(y[right_indices])
                n_l, n_r = len(left_indices), len(right_indices)
                child_entropy = (n_l / num_samples) * e_left + (n_r / num_samples) * e_right
                gain = current_entropy - child_entropy

                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_idx, threshold, left_indices, right_indices)

        if best_gain > 0:
            idx, thr, left_idx, right_idx = best_split
            left_subtree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
            right_subtree = self._build_tree(X[right_idx], y[right_idx], depth + 1)
            return {"feature": idx, "threshold": thr, "left": left_subtree, "right": right_subtree}
        
        return np.bincount(y).argmax()

    def predict(self, x):
        """予測の入り口"""
        return self._predict_recursive(x, self.tree)

    def _predict_recursive(self, x, node):
        """【再帰】構築された木を辿って予測を返す"""
        if not isinstance(node, dict):
            return node
        
        feature_val = x[node['feature']]
        if feature_val <= node['threshold']:
            return self._predict_recursive(x, node['left'])
        else:
            return self._predict_recursive(x, node['right'])

# --- 実行セクション ---

# 訓練データ: [カサが滑らかか(0 or 1), 匂いがあるか(0 or 1)]
X_train = np.array([
    [0, 0], [0, 1], [1, 0], [1, 1], [0, 1], [1, 1]
])
y_train = np.array([0, 1, 0, 1, 1, 1])  # 0:食用, 1:毒

# 学習
model = DecisionTree(max_depth=2)
model.fit(X_train, y_train)

print("--- 学習完了 ---")
print("構築された決定木（辞書構造）:", model.tree)

# 予測のテスト
val1 = ask_question("カサは滑らかですか？(yes or no)")
val2 = ask_question("匂いはありますか？(yes or no)")
test_mushroom = np.array([val1, val2])

result = model.predict(test_mushroom)

print("\n--- 予測テスト ---")
print(f"特徴{test_mushroom} のキノコは...")
print(f"結果: {'⚠️ 毒キノコです！' if result == 1 else '🍄 食用キノコです。'}")