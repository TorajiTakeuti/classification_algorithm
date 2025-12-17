import numpy as np
# 1. 情報エントロピーの計算
def calculate_entropy(y):
    if len(y) == 0:
        return 0
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

# 2. 決定木アルゴリズム
class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """学習（行列Xとラベルyを使用して木を構築）"""
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        """【再帰】学習プロセス（ログ出力付き）"""
        # --- 思考ログの追加 ---
        indent = "  " * depth  # 深さに応じて字下げ
        print(f"{indent}[深さ {depth}] 学習開始 - サンプル数: {len(y)}")
        
        num_samples, num_features = X.shape
        
        # 停止条件のチェック
        if len(np.unique(y)) == 1:
            label = "毒" if y[0] == 1 else "食用"
            print(f"{indent}  => 全て同じ種類 ({label}) のため、この枝は終了")
            return np.bincount(y).argmax()
            
        if depth >= self.max_depth:
            print(f"{indent}  => 最大深度に達したため、多数決で終了")
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
            feat_name = "カサ" if idx == 0 else "匂い"
            print(f"{indent}  [決定] 特徴量 '{feat_name}' で分割 (情報利得: {best_gain:.4f})")
            
            # 再帰呼び出し（子ノードへ）
            left_subtree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
            right_subtree = self._build_tree(X[right_idx], y[right_idx], depth + 1)
            
            return {"feature": idx, "threshold": thr, "left": left_subtree, "right": right_subtree}
        
        return np.bincount(y).argmax()

    def predict(self, x):
        """予測の入り口"""
        return self._predict_recursive(x, self.tree)

    def _predict_recursive(self, x, node, depth=0):
        indent = "  " * depth
        if not isinstance(node, dict):
            label = "毒" if node == 1 else "食用"
            print(f"{indent}結論: このキノコは【{label}】です")
            return node
        
        feat_name = "カサ" if node['feature'] == 0 else "匂い"
        val = "あり(1)" if x[node['feature']] == 1 else "なし(0)"
        print(f"{indent}質問: {feat_name} は {val} ですか？")
        
        if x[node['feature']] <= node['threshold']:
            print(f"{indent}  -> [左の枝へ]")
            return self._predict_recursive(x, node['left'], depth + 1)
        else:
            print(f"{indent}  -> [右の枝へ]")
            return self._predict_recursive(x, node['right'], depth + 1)
