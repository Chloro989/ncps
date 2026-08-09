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

# 毎回まっさらにクローンし直す。
# 古いクローンを更新する方式にしていたら、前回の出力が残って
# 実行していない結果を持ち帰る事故が2回起きた。作り直すほうが安全で速い
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run(['git', 'clone', '-q',
                'https://github.com/Chloro989/ncps', str(REPO)], check=True)

subprocess.run(['pip', 'install', '-q', 'ncps'], check=True)

import os
os.chdir(WORK)
print('作業場所:', WORK)
print('コミット:', subprocess.run(
    ['git', 'log', '--oneline', '-1'],
    capture_output=True, text=True).stdout.strip())
```

学習に必要な入力(`centurion_trace.npz` と `centurion_trace.txt`)は
**リポジトリに入っている**ので、Drive から戻す必要はない。
Drive は成果物を持ち出すためだけに使う。

前回の続きから学習を再開したいときだけ、次を追加で実行する。

```python
shutil.copy2(STORE / 'centurion_circuit.pt', WORK / 'centurion_circuit.pt')
```

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

# 実行して作られたものだけを保存する。入力ファイルはリポジトリにあるので触らない
saved = []
for name in ('centurion_train.txt', 'centurion_circuit.pt',
             'centurion_samples.txt', 'centurion_fluency.npz'):
    path = WORK / name
    if path.exists():
        shutil.copy2(path, STORE / name)
        saved.append(name)
print('保存:', saved)

# 持ち帰る前に、それが今回の実行のものか確かめる
log = WORK / 'centurion_train.txt'
if log.exists():
    print('\n--- ログの先頭 ---')
    print('\n'.join(log.read_text(encoding='utf-8').splitlines()[:5]))
```

最後に表示される「版」と「開始」が今回の実行のものであることを確認してから
ダウンロードする。ここが前回と同じなら、学習は走っていない。

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
