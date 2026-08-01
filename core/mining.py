import numpy as np
import pandas as pd

from core.data_loader import df
from core.constants import BASE_COLS, FEATURE_LABELS_ID

def build_transaction_df(data):
    RULE_COLS = BASE_COLS + ["Exited"]

    trans = pd.DataFrame({
        c: data[c + "_label"] if (c + "_label") in data.columns else data[c].astype(str)
        for c in RULE_COLS
    })

    for c in ["Tenure", "NumOfProducts"]:
        trans[c] = FEATURE_LABELS_ID[c] + "=" + data[c].astype(str)

    for c in RULE_COLS:
        if c not in ["Tenure", "NumOfProducts"]:
            trans[c] = FEATURE_LABELS_ID.get(c, c) + "=" + trans[c].astype(str)

    return trans



def _get_encoded_items(data):
    """Mengubah dataframe menjadi matriks one-hot untuk Apriori."""
    
    trans = build_transaction_df(data)

    one_hot = pd.get_dummies(
        trans,
        prefix={c: "" for c in trans.columns},
        prefix_sep=""
    )

    return one_hot.astype(bool)


def mine_association_rules(data, min_support=0.10, min_confidence=0.6, max_len=2):
    """Mining aturan asosiasi 1-ke-1 (single item -> single item) agar mudah
    divisualisasikan sebagai jaringan aturan (rule network)."""
    try:
        from mlxtend.frequent_patterns import apriori, association_rules

        one_hot = _get_encoded_items(data)
        freq = apriori(one_hot, min_support=min_support, use_colnames=True, max_len=max_len)
        if freq.empty:
            return pd.DataFrame(columns=["antecedent", "consequent", "support", "confidence", "lift"])
        rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
        rules = rules[(rules["antecedents"].apply(len) == 1) & (rules["consequents"].apply(len) == 1)]
        if rules.empty:
            return pd.DataFrame(columns=["antecedent", "consequent", "support", "confidence", "lift"])
        rules = rules.copy()
        rules["antecedent"] = rules["antecedents"].apply(lambda s: list(s)[0])
        rules["consequent"] = rules["consequents"].apply(lambda s: list(s)[0])
        rules = rules[["antecedent", "consequent", "support", "confidence", "lift"]]
        rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
        return rules
    except ImportError:
        return _mine_rules_fallback(
            data,
            min_support,
            min_confidence
        )


def _mine_rules_fallback(data, min_support, min_confidence):
    """Mesin cadangan (tanpa mlxtend): pairwise apriori manual."""
    one_hot = _get_encoded_items(data)
    n = len(one_hot)
    item_arrays = {c: one_hot[c].values for c in one_hot.columns}

    supports = {}
    for name, arr in item_arrays.items():
        s = arr.mean()
        if s >= min_support:
            supports[name] = s

    freq_items = list(supports.keys())
    rules = []
    for i in range(len(freq_items)):
        for j in range(len(freq_items)):
            if i == j:
                continue
            a, b = freq_items[i], freq_items[j]
            if a.split("=")[0] == b.split("=")[0]:
                continue
            sup_ab = (item_arrays[a] & item_arrays[b]).mean()
            if sup_ab < min_support:
                continue
            conf = sup_ab / supports[a]
            if conf < min_confidence:
                continue
            lift = sup_ab / (supports[a] * supports[b])
            rules.append((a, b, sup_ab, conf, lift))

    rules_df = pd.DataFrame(rules, columns=["antecedent", "consequent", "support", "confidence", "lift"])
    if not rules_df.empty:
        rules_df = rules_df.sort_values("lift", ascending=False).reset_index(drop=True)
    return rules_df


# Precompute default rule set untuk narasi Knowledge Discovery Report
_default_rules = mine_association_rules(
    df,
    min_support=0.10,
    min_confidence=0.6
)


def top_rule_involving(keyword, rules_df=_default_rules, n=1):
    subset = rules_df[rules_df["consequent"].str.contains(keyword, case=False, na=False)]
    return subset.head(n)

