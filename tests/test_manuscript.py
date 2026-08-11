"""
原稿の読み取りの試験。

    python tests/test_manuscript.py

モデルを使わないので手元のCPUだけで全部動く。
一番大事なのは**位置が合っているか**で、ここがずれると
「三章の二段落目」という指摘そのものが嘘になる。
"""

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from centurion.manuscript import (Manuscript, clean, split_sentences,
                                  strip_aozora, width)

FIXTURES = HERE / "fixtures"
passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ○ {name}")
    else:
        failed += 1
        print(f"  × {name}" + (f" — {detail}" if detail else ""))


def offsets_match(manuscript):
    """段落が持つ位置が、本文の実際の位置と一致するか"""
    return all(manuscript.text[p.start:p.end] == p.text
               for p in manuscript.paragraphs)


print("== 自作の原稿 (空行で区切る書き方) ==")
novel = Manuscript.load(FIXTURES / "sample_novel.txt")
check("マークダウンの見出しを題にする", novel.title == "試験用の原稿",
      novel.title)
check("章を3つ見つける", len(novel.chapters) == 3, str(len(novel.chapters)))
check("章の見出しを拾う",
      [c.title for c in novel.chapters] == ["第一章", "第二章", "＊"],
      str([c.title for c in novel.chapters]))
check("見出しは段落に含めない",
      not any("第一章" == p.text for p in novel.paragraphs))
check("位置が全部合っている", offsets_match(novel))
check("会話文が独立した段落になる",
      any(p.text == "「遅くなった」" for p in novel.paragraphs))
check("心内文も段落になる",
      any(p.text.startswith("（") for p in novel.paragraphs))
check("ルビを落とす", "《" not in novel.text and "鳥の声" in novel.text)
check("入力者注を落とす", "［＃" not in novel.text)
check("章ごとに段落が属する",
      all(p.chapter == c.index for c in novel.chapters
          for p in c.paragraphs))

print("\n== 青空文庫 (1行1段落、付属物つき) ==")
aozora = Manuscript.load(FIXTURES / "aozora_akuma.txt")
check("題を拾う", aozora.title == "悪魔", aozora.title)
check("著者を拾う", aozora.author == "芥川龍之介", aozora.author)
check("凡例を落とす", "【テキスト中に現れる記号について】" not in aozora.text)
check("底本の記録を落とす", "底本" not in aozora.text)
check("青空文庫の案内を落とす", "青空文庫" not in aozora.text)
check("ルビを落として親字は残す",
      "《" not in aozora.text and "伴天連うるがん" in aozora.text)
check("傍点の指定を落とす", "傍点" not in aozora.text)
check("本文が残っている", "私は寂しくつて仕方がありません" in aozora.text)
check("空行が無くても段落に分かれる", len(aozora.paragraphs) >= 10,
      str(len(aozora.paragraphs)))
check("位置が全部合っている", offsets_match(aozora))

print("\n== 文に分ける ==")
check("句点で切る",
      split_sentences("雨が降った。傘をさした。") == ["雨が降った。", "傘をさした。"])
check("閉じ括弧まで含めて一文にする",
      split_sentences("「行こう。」と言った。")
      == ["「行こう。」", "と言った。"])
check("会話が続いても混ざらない",
      split_sentences("「乗り遅れた」と空は言った。「二十分に一本だ」")
      == ["「乗り遅れた」と空は言った。", "「二十分に一本だ」"])
check("句点で終わらない行も一文として拾う",
      split_sentences("「客、来ねえの」") == ["「客、来ねえの」"])
check("感嘆符と疑問符でも切る",
      len(split_sentences("え? そうか! なるほど。")) == 3)
check("空文字なら何も返さない", split_sentences("") == [])

print("\n== 見た目の長さ ==")
check("全角は2、半角は1", width("あab") == 4, str(width("あab")))
check("空文字は0", width("") == 0)

print("\n== 切り分け ==")
chunks = novel.chunks(size=600, overlap=1)
check("章をまたがない",
      all(len({p.chapter for p in c.paragraphs}) == 1 for c in chunks))
check("章の数だけ塊ができる", len(chunks) == len(novel.chapters),
      str(len(chunks)))
check("章をまたぐときは持ち越さない",
      all(c.carried == 0 for c in chunks))

small = aozora.chunks(size=500, overlap=1)
check("上限を超えたら分ける", len(small) > 1, str(len(small)))
check("2つ目以降は前の段落を持ち越す",
      all(c.carried == 1 for c in small[1:]),
      str([c.carried for c in small]))
check("持ち越した段落は前の塊の末尾と同じ",
      all(small[i].paragraphs[0].index == small[i - 1].paragraphs[-1].index
          for i in range(1, len(small))))
check("担当範囲は重ならない",
      all(small[i].span[0] > small[i - 1].span[1]
          for i in range(1, len(small))))
check("全段落がどこかの担当になる",
      sorted(p.index for c in small
             for p in c.paragraphs[c.carried:])
      == [p.index for p in aozora.paragraphs])
check("重なりを除いた本文が取れる",
      small[1].body and small[1].body != small[1].text)

huge = aozora.chunks(size=10, overlap=0)
check("1段落で上限を超えてもその段落は捨てない",
      sum(len(c.paragraphs) for c in huge) == len(aozora.paragraphs),
      f"{sum(len(c.paragraphs) for c in huge)} 対 {len(aozora.paragraphs)}")

by_token = novel.chunks(size=120, measure=lambda t: len(t) // 2)
check("大きさの測り方を差し替えられる", len(by_token) >= len(chunks))

print("\n== 文字コード ==")
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "shift_jis.txt"
    path.write_bytes("　雪が降った。\n\n「寒いね」\n".encode("cp932"))
    shifted = Manuscript.load(path)
    check("Shift_JIS を読める", "雪が降った" in shifted.text)
    check("段落に分かれる", len(shifted.paragraphs) == 2,
          str(len(shifted.paragraphs)))

print("\n== 端の場合 ==")
empty = Manuscript("")
check("空の原稿でも落ちない",
      empty.paragraphs == [] and empty.chapters == []
      and empty.chunks() == [])
check("見出しだけの原稿でも落ちない", Manuscript("第一章").paragraphs == [])
check("改行の書き方が違っても揃う",
      clean("あ\r\nい\rう") == "あ\nい\nう")

print(f"\n{passed}件通過 / {failed}件失敗")
raise SystemExit(1 if failed else 0)
