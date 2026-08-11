# 原稿の置き場

自分の原稿をここに置く。**この中身は git に入らない**
(`.gitignore` で全部除いてある)。未発表の作品を公開リポジトリへ
うっかり入れないための場所なので、除外の設定は消さないこと。

ここに置いたものは、ファイル名だけで呼べる。

```bash
python -m centurion.critique 第五稿.txt --list
```

パスを省くと窓が開く。手元のPCならファイル選択、Colab ならアップロード。

```bash
python -m centurion.critique --list
```

Colab からPCのファイルを受け取るときは、これでもよい。

```python
from centurion.manuscript import Manuscript
manuscript = Manuscript.load()      # アップロードの窓が出る
print(manuscript.summary())
```

試験に使う見本は、ここではなく `tests/fixtures/` にある
(自作のものと、著作権の切れたもの)。
