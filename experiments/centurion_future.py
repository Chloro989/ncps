"""
センチュリオン: 未来エントロピーによる重み付け (Phase 6)

抑圧は失敗した。Phase 5 で、抑圧を強めるほど轍語彙が増えることが
測定で確定している(相関+0.862)。貪欲法の1位はしばしば機能語で、
それを押し下げると内容語が繰り上がる。この領域の高確率な内容語とは
深淵・神秘・星そのもので、抑圧は轍の語を優先的に引き上げていた。

未来エントロピーは別の機構をとる。候補を押し下げるのではなく、
各候補について「それを選んだ次の分布がどれだけ開くか」を測り、
先が開ける候補を重くする。

  s(w) = p(w) * Ĥ(w)^α        Ĥ(w) = 候補wの次の分布の正規化エントロピー

轍の語は定型句なので、選んだ後の展開が限られる。つまり Ĥ が小さく、
自然に下がる。明示的な門は要らない — 文法的に決まっている箇所では
1位の確率がほぼ1で、他候補を持ち上げても順位が覆らないため、
この機構は自己ゲート的に働く。

計算量の要はKVキャッシュにある。素直に書くと毎ステップ全文脈を
n本ぶん流し直すことになり100倍近くなるが、キャッシュを
バッチ方向に広げて1トークンだけ進めれば、追加は1ステップ2回の順伝播で済む。
"""

import torch

TOP_N = 6              # 先読みする候補数
ALPHA = 1.0            # 未来エントロピーの効き。0で無効、1で記事の式そのもの
MIN_P = 0.05           # 候補の足切り。既存の設定に合わせる


def normalized_entropy(probs, top_n):
    """上位top_nに絞った分布のエントロピーを、0〜1に正規化して返す"""
    values = probs.topk(top_n, dim=-1).values
    values = values / values.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    entropy = -(values * torch.log(values + 1e-9)).sum(dim=-1)
    return entropy / torch.log(torch.tensor(float(top_n), device=probs.device))


def lookahead_entropy(model, cache, candidates, position):
    """各候補を1つ進めたときの、次の分布の正規化エントロピーを返す。

    キャッシュをバッチ方向に広げて1トークンだけ進め、
    終わったら元の形に戻す。コピーを作らないのが要点"""
    count = candidates.numel()
    length = cache.get_seq_length()

    cache.batch_repeat_interleave(count)
    try:
        with torch.no_grad():
            output = model(
                input_ids=candidates.view(count, 1),
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.tensor([position], device=candidates.device),
            )
        probs = torch.softmax(output.logits[:, -1, :].float(), dim=-1)
        return normalized_entropy(probs, TOP_N)
    finally:
        # 広げた分と、進めた1トークン分を戻す。
        # 全行が同じ履歴を持っていたので、0行目を残せば元の状態になる
        cache.batch_select_indices(torch.tensor([0], device=candidates.device))
        cache.crop(length)


def reweight(probs, model, cache, position, top_n=TOP_N, alpha=ALPHA,
             min_p=MIN_P):
    """未来エントロピーで重み付けした分布を返す。
    戻り値は (候補のトークンID, 重み付け後の確率)"""
    values, indices = probs.topk(top_n)

    # min_p の足切りは既存の設定と揃える。ただし最低1つは残す
    keep = values >= min_p * values[0]
    keep[0] = True
    values, indices = values[keep], indices[keep]

    if indices.numel() == 1 or alpha == 0.0:
        return indices, values / values.sum()

    future = lookahead_entropy(model, cache, indices, position)
    weighted = values * future.clamp(min=1e-6) ** alpha
    total = weighted.sum()
    if total <= 0:
        return indices, values / values.sum()
    return indices, weighted / total


def generate(model, tokenizer, input_ids, max_new_tokens,
             greedy=False, top_n=TOP_N, alpha=ALPHA, min_p=MIN_P,
             collect=None):
    """未来エントロピーで重み付けしながら1トークンずつ生成する。
    collect にリストを渡すと、各ステップの (未来エントロピーの幅, 選んだ順位) が入る"""
    device = input_ids.device
    with torch.no_grad():
        output = model(input_ids=input_ids, use_cache=True)
    cache = output.past_key_values
    logits = output.logits[0, -1, :].float()

    generated = []
    position = input_ids.shape[1]

    for _ in range(max_new_tokens):
        probs = torch.softmax(logits, dim=-1)
        indices, weights = reweight(probs, model, cache, position,
                                    top_n, alpha, min_p)

        if greedy:
            choice = int(weights.argmax())
        else:
            choice = int(torch.multinomial(weights, 1))
        token = indices[choice]

        if collect is not None:
            # 元の確率での順位。0なら1位を選んでいる = 何も変えていない
            collect.append((float(weights.max() - weights.min()), choice))

        if token.item() == tokenizer.eos_token_id:
            break
        generated.append(token.item())

        with torch.no_grad():
            output = model(
                input_ids=token.view(1, 1),
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.tensor([position], device=device),
            )
        cache = output.past_key_values
        logits = output.logits[0, -1, :].float()
        position += 1

    return generated
