import math
import re
from collections import Counter


def _tokenize(s: str) -> list[str]:
    return re.findall(r'\S+', s)


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def bleu4(hypotheses: list[str], references: list[str]) -> float:
    """Corpus-level BLEU-4 with brevity penalty."""
    total_hyp_len = 0
    total_ref_len = 0
    clipped = [0] * 4
    total   = [0] * 4

    for hyp, ref in zip(hypotheses, references):
        hyp_tok = _tokenize(hyp)
        ref_tok = _tokenize(ref)
        total_hyp_len += len(hyp_tok)
        total_ref_len += len(ref_tok)

        for n in range(1, 5):
            hyp_ng = _ngram_counts(hyp_tok, n)
            ref_ng = _ngram_counts(ref_tok, n)
            for gram, cnt in hyp_ng.items():
                clipped[n-1] += min(cnt, ref_ng.get(gram, 0))
            total[n-1] += max(0, len(hyp_tok) - n + 1)

    precisions = []
    for c, t in zip(clipped, total):
        if t == 0:
            return 0.0
        precisions.append(math.log(c / t) if c > 0 else float("-inf"))

    if any(math.isinf(p) for p in precisions):
        return 0.0

    bp = min(1.0, math.exp(1 - total_ref_len / max(total_hyp_len, 1)))
    return bp * math.exp(sum(precisions) / 4) * 100


def exact_match(hypotheses: list[str], references: list[str]) -> float:
    """Percentage of predictions that exactly match the reference after stripping whitespace."""
    correct = sum(1 for h, r in zip(hypotheses, references) if h.strip() == r.strip())
    return correct / max(len(hypotheses), 1) * 100


def edit_distance(hypotheses: list[str], references: list[str]) -> float:
    """Mean normalised edit distance (token-level). Lower is better."""
    def _levenshtein(a: list, b: list) -> int:
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if a[i-1] == b[j-1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j-1])
                prev = temp
        return dp[n]

    scores = []
    for h, r in zip(hypotheses, references):
        h_tok = _tokenize(h)
        r_tok = _tokenize(r)
        d = _levenshtein(h_tok, r_tok)
        scores.append(d / max(len(h_tok), len(r_tok), 1))
    return sum(scores) / max(len(scores), 1)


def compute_metrics(hypotheses: list[str], references: list[str]) -> dict:
    return {
        "bleu4":      round(bleu4(hypotheses, references), 4),
        "exact_match": round(exact_match(hypotheses, references), 4),
        "edit_dist":  round(edit_distance(hypotheses, references), 4),
    }
