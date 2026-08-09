# Colab の使い方

ランタイムが切れるとファイルは消える。消えて困るものは2種類あり、置き場所を分ける。

- **コード** — GitHub に置く。毎回クローンし直せばよい
- **成果物**(トレース、標本、学習済みパラメータ)— Google Drive に置く。作り直すのに時間がかかる

## 最初のセル(毎回これを実行する)

```python
# ===== センチュリオン: Colab の準備 =====
from google.colab import drive
drive.mount('/content/drive')

import pathlib, shutil, subprocess

STORE = pathlib.Path('/content/drive/MyDrive/centurion')   # 成果物の置き場
REPO = pathlib.Path('/content/ncps')                       # コード
WORK = REPO / 'experiments'
STORE.mkdir(parents=True, exist_ok=True)

# コードを取得。2回目以降は最新に更新するだけ
if (REPO / '.git').exists():
    subprocess.run(['git', '-C', str(REPO), 'pull', '-q'], check=True)
else:
    subprocess.run(['git', 'clone', '-q',
                    'https://github.com/Chloro989/ncps', str(REPO)], check=True)

# 前回までの成果物を作業場所へ戻す
restored = []
for path in STORE.iterdir():
    if path.is_file():
        shutil.copy2(path, WORK / path.name)
        restored.append(path.name)

subprocess.run(['pip', 'install', '-q', 'ncps'], check=True)

import os
os.chdir(WORK)
print('作業場所:', WORK)
print('復元:', restored or 'なし(初回)')

# どの版のコードを使うのかを必ず確認する。
# 古いクローンのまま走らせて、結果を取り違えたことがある
print('コミット:', subprocess.run(
    ['git', '-C', str(REPO), 'log', '--oneline', '-1'],
    capture_output=True, text=True).stdout.strip())
```

**復元されたログとチェックポイントは前回の実行のもの**である点に注意する。
学習を回さずに持ち帰ると、前回と同じファイルを渡すことになる。
`centurion_train.txt` の先頭には版と開始時刻が書かれているので、
持ち帰る前にそこを見て、今回の実行のものか確かめること。

## 実行

```python
!python prot_type7_train.py
```

## 最後のセル(必ず実行する)

これを忘れるとランタイム終了で成果物が消える。

```python
# ===== 成果物を Drive に保存 =====
import pathlib, shutil

STORE = pathlib.Path('/content/drive/MyDrive/centurion')
WORK = pathlib.Path('/content/ncps/experiments')

saved = []
for pattern in ('centurion_*.npz', 'centurion_*.pt',
                'centurion_train.txt', 'centurion_samples.txt',
                'centurion_trace.txt', 'centurion_fluency.npz'):
    for path in WORK.glob(pattern):
        shutil.copy2(path, STORE / path.name)
        saved.append(path.name)
print('保存:', sorted(set(saved)))
```

## 長い学習を回すとき

学習の途中でランタイムが切れると、そこまでの計算が消える。
`prot_type7_train.py` は5世代ごとに `centurion_circuit.pt` を書くので、
保存先を Drive に向けておけば途中で切れても続きから始められる。

```python
!python prot_type7_train.py
```
の代わりに、チェックポイントを直接 Drive に置く場合は
`CHECKPOINT` を `/content/drive/MyDrive/centurion/centurion_circuit.pt` に変える。

## 成果物の一覧

| ファイル | 作るもの | 用途 |
|---|---|---|
| `centurion_trace.npz` | `prot_type7_trace.py` | 特徴の標準化統計。回路が必ず読む |
| `centurion_trace.txt` | 同上 | 轍の重心。報酬が必ず読む |
| `centurion_samples.txt` | `prot_type7_sample.py` | ラベル付け用の標本 |
| `centurion_fluency.npz` | `prot_type7_fluency.py` | 尤度の実測 |
| `centurion_circuit.pt` | `prot_type7_train.py` | 学習済みパラメータ |
| `centurion_train.txt` | 同上 | 学習のログ |

上2つは作り直すのに生成20回分かかる。消さないこと。
