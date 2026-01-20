import numpy as np

def calculate_entropy(y):
    """情報エントロピーの計算（ベクトル演算で高速化）"""
    if len(y) == 0: return 0
    probabilities = np.bincount(y) / len(y)
    # 0の時にlog計算を避けるため、正の値のみ抽出
    p = probabilities[probabilities > 0]
    return -np.sum(p * np.log2(p))

def calculate_gini(y):
    """ジニ不純度の計算（エントロピーより計算が軽く、実用性が高い）"""
    if len(y) == 0: return 0
    probabilities = np.bincount(y) / len(y)
    return 1 - np.sum(probabilities**2)

class DecisionTree:
    def __init__(self, max_depth=3, criterion='entropy', feature_names=None):
        self.max_depth = max_depth
        self.criterion = criterion
        self.feature_names = feature_names
        self.tree = None
        
        # 指標の切り替え
        self.metric = calculate_entropy if criterion == 'entropy' else calculate_gini

    def fit(self, X, y):
        # 特徴量名が未指定の場合は「特徴量0, 1...」とする
        if self.feature_names is None:
            self.feature_names = [f"特徴量{i}" for i in range(X.shape[1])]
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        
        # --- 停止条件 ---
        # 1. 全て同じラベル
        if len(np.unique(y)) == 1:
            return y[0]
        # 2. 最大深度に達した、またはサンプルが少なすぎる
        if depth >= self.max_depth or num_samples < 2:
            return np.bincount(y).argmax()

        # --- 最良の分割を探す ---
        best_gain = -1
        best_split = None
        current_impurity = self.metric(y)

        for idx in range(num_features):
            thresholds = np.unique(X[:, idx])
            for thr in thresholds:
                # データを分割
                left_mask = X[:, idx] <= thr
                right_mask = ~left_mask # 反転ビット演算で高速化
                
                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                # 重み付き不純度の計算
                n_l, n_r = np.sum(left_mask), np.sum(right_mask)
                w_l, w_r = n_l / num_samples, n_r / num_samples
                child_impurity = w_l * self.metric(y[left_mask]) + w_r * self.metric(y[right_mask])
                
                gain = current_impurity - child_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_split = (idx, thr, left_mask, right_mask)

        # --- 分割の実行 ---
        if best_gain > 0:
            idx, thr, left_m, right_m = best_split
            
            # 再帰的に子ノードを構築
            left_subtree = self._build_tree(X[left_m], y[left_m], depth + 1)
            right_subtree = self._build_tree(X[right_m], y[right_m], depth + 1)
            
            return {
                "feature_idx": idx,
                "feature_name": self.feature_names[idx],
                "threshold": thr,
                "left": left_subtree,
                "right": right_subtree,
                "gain": best_gain
            }
        
        return np.bincount(y).argmax()

    def predict(self, X):
        """複数サンプル（行列）の一括予測にも対応"""
        if X.ndim == 1:
            return self._traverse(X, self.tree)
        return np.array([self._traverse(x, self.tree) for x in X])

    def _traverse(self, x, node):
        if not isinstance(node, dict):
            return node
        
        if x[node['feature_idx']] <= node['threshold']:
            return self._traverse(x, node['left'])
        else:
            return self._traverse(x, node['right'])