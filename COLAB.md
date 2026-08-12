# Colab の使い方

## センチュリオンを動かすだけなら

実験ではなく製品を使うだけなら、これで足りる。Drive も要らない。

```python
!git clone https://github.com/Chloro989/ncps /content/ncps
%cd /content/ncps
```

### 原稿を読ませる (GPU不要)

`read` `ask` `check` は GPU を使わないので、clone しただけで動く。
`ask` が出したものを Claude なり ChatGPT なりに貼れば、性能の高いモデルで読ませられる。

その場で論評まで出すなら、鍵を置いてから `--api` を付ける。

```python
import os
os.environ['ANTHROPIC_API_KEY'] = '自分の鍵'     # console.anthropic.com で作る
```

```python
!python main.py ask 第五稿.txt --mode 発想 --api --all
```

**まずセルの中でアップロードする。** 窓は IPython の kernel を通して開くので、
`!python` で起動した子プロセスからは開けない。

```python
import sys; sys.path.insert(0, '/content/ncps')
from centurion.manuscript import upload

upload()        # 窓が出る。選んだファイルは manuscripts/ に入る
```

置いたあとは、`!python` からファイル名で呼べる。

```python
!python main.py read 第五稿.txt
!python main.py ask 第五稿.txt --mode 接続 --dream
```

原稿が一つだけなら名前も省ける。

```python
!python main.py read
```

セルの中だけで済ませることもできる。こちらは窓が開くので、
アップロードから読ませるまで一息でできる。

```python
import main
main.main(["read"])                 # アップロードの窓が出る
main.main(["ask", "--mode", "接続"])
```

`Manuscript` を直に使う場合も同じ。

```python
from centurion.manuscript import Manuscript

manuscript = Manuscript.load()      # アップロードの窓が出る
print(manuscript.summary())
```

ランタイムが切れると `manuscripts/` も消える。何度も使う原稿は Drive に置く。

```python
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/原稿/*.txt /content/ncps/manuscripts/
```

### 小説を書かせる (モデルが要る)

```python
!pip install -q transformers accelerate
```

```python
!python main.py write 誰かが置いていった傘の話をして --verbose
```

Python から使うなら、モデルの読み込みに数分かかるので
`Centurion()` は一度だけ作って使い回すこと。

```python
from centurion import Centurion

centurion = Centurion()
for reply in centurion.converse(["朝の匂いについて書いて",
                                 "沈黙について書いて"]):
    print(reply, "\n")
```

以下は実験を回すときの手順。

## 実験のとき

ランタイムが切れるとファイルは消える。消えて困るものは2種類あり、置き場所を分ける。

- **コード** — GitHub に置く。毎回クローンし直せばよい
- **成果物**(トレース、標本、学習済みパラメータ)— Google Drive に置く。作り直すのに時間がかかる

## 最初のセル(毎回これを実行する)

```python
# ===== センチュリオン: Colab の準備 =====
from google.colab import drive
drive.mount('/content/drive')

import os, pathlib, shutil, subprocess

STORE = pathlib.Path('/content/drive/MyDrive/centurion')   # 成果物の置き場
REPO = pathlib.Path('/content/ncps')                       # コード
WORK = REPO / 'experiments'
STORE.mkdir(parents=True, exist_ok=True)

# 消す前に、消す対象の外へ出ておく。
# 作業場所の中に居たまま消すとカレントディレクトリが無効になり、
# 直後の git が exit 128 で落ちる(セルを2回目に実行したときに必ず起きる)
os.chdir('/content')

# 毎回まっさらにクローンし直す。
# 古いクローンを更新する方式にしていたら、前回の出力が残って
# 実行していない結果を持ち帰る事故が2回起きた。作り直すほうが安全で速い
if REPO.exists():
    shutil.rmtree(REPO)

# -q は使わない。失敗したときに理由が見えなくなる
result = subprocess.run(
    ['git', 'clone', 'https://github.com/Chloro989/ncps', str(REPO)],
    capture_output=True, text=True)
if result.returncode != 0:
    raise SystemExit('clone に失敗:\n' + result.stderr)

subprocess.run(['pip', 'install', '-q', 'ncps'], check=True)

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
