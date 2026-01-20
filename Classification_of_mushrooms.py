import numpy as np
import sys
from decisionTree import DecisionTree

def ask_question(prompt):
    """質問をして 1(yes) か 0(no) を返す。それ以外は終了。"""
    response_map = {
        "yes": 1, "y": 1, "はい": 1, "ハイ": 1, "ok": 1,"True":1,
        "no": 0,  "n": 0, "いいえ": 0, "イイエ": 0, "ダメ": 0, "False":0
    }
    answer = input(prompt).lower() # 小文字に統一して判定
    if answer in response_map:
        return response_map[answer]
    else:
        print(f'{answer}は理解できません。最初からやり直してください')
        sys.exit()

# --- 実行セクション ---

# 訓練データ: [カサが滑らかか(0 or 1), 匂いがあるか(0 or 1)]
X_train = np.array([
    [0, 0], [0, 1], [1, 0], [1, 1], [0, 1], [1, 1]
])
y_train = np.array([0, 1, 0, 1, 1, 1])  # 0:食用, 1:毒

# 学習
model = DecisionTree(max_depth=3)
model.fit(X_train, y_train)

print("--- 学習完了 ---")
print("構築された決定木（辞書構造）:", model.tree)

# 予測のテスト
val1 = ask_question("カサは滑らかですか？")
val2 = ask_question("匂いはありますか？")
test_mushroom = np.array([val1, val2])

result = model.predict(test_mushroom)

print("\n--- 予測テスト ---")

user_input = [val1, val2]
if user_input == ([0, 0]):
    feature = "カサが滑らかでなく匂いがない"
elif user_input == ([1, 0]):
    feature = "カサが滑らかで匂いがない"
elif user_input == ([0, 1]):
    feature = "カサが滑らかでなく匂いがある"
else :
    feature = "カサが滑らかで匂いがある"

print(f"特徴:{feature}キノコは...")
print(f"結果: {'⚠️ 毒キノコです！' if result == 1 else '🍄 食用キノコです。'}")