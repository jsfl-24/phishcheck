import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = ROOT / "feature_columns.json"
DATA_URL = "hf://datasets/pirocheto/phishing-url/data/train.parquet"

_S2_LABELS = {
    "com", "co", "org", "net", "gov", "go", "ac", "edu", "ed", "or", "web", "my",
    "sch", "mil", "gob", "gen", "ind", "firm", "ne", "lg", "k12", "nic", "com1",
}
_SUS_S2_KEYS = [
    "ae", "asia", "audio", "bj", "club", "exposed", "ga", "go.id", "gob.pe",
    "icu", "la", "link", "media", "ml", "mx", "party", "pe", "review", "rs",
    "tk", "top", "website", "work", "xyz",
]
_SHORTENERS = [
    "bit.ly", "goo.gl", "shorte.st", "x.co", "ow.ly", "t.co", "tinyurl.com",
    "tr.im", "is.gd", "feeds.feedburner", "adf.ly", "bit.do", "tiny.cc", "url.ie",
    "lycos.com", "bitly.com", "cur.tt", "snipurl.com", "short.to", "budurl.com",
    "ping.fm", "post.ly", "just.as", "bkite.com", "snipr.com", "fic.kr",
    "loopt.us", "doiop.com", "short.ie", "kl.am", "twitthis.com", "u.to",
    "j.mp", "buzurl.com", "cutt.us", "yourls.org", "prettylinkpro.com",
    "scrnch.me", "vurl.bz", "v.gd", "link.zip.net", "buff.ly", "lnkd.in",
    "rb.gy", "cutt.ly", "shorturl.at", "rebrand.ly",
]
_HINTS_A = [
    "wp", "login", "includes", "admin", "content", "site", "images", "js",
    "alibaba", "css", "myaccount", "dropbox", "themes", "plugins", "signin",
    "view",
]
_HINTS_B = [
    "login", "signin", "secure", "account", "verify", "update", "password",
    "confirm", "banking", "payment", "free", "bonus", "wallet", "ebay",
    "paypal", "webscr", "bank", "passwords", "auth", "session", "unlock",
    "recover", "validate",
]
_SPLIT_NONALNUM = r"[^A-Za-z0-9]"
_SPLIT_URL = r"[-._/?=&~%+#]"
_SPLIT_PCT = r"[-._/?=&:@%#~]"
_IPV4 = re.compile(r"\d{1,3}(\.\d{1,3}){3}")
_PORT = re.compile(r":\d+$")


def load_split(name="train"):
    try:
        return pd.read_parquet(f"hf://datasets/pirocheto/phishing-url/data/{name}.parquet")
    except Exception:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="pirocheto/phishing-url",
            filename=f"data/{name}.parquet",
            repo_type="dataset",
        )
        return pd.read_parquet(path)


def _parse(url):
    raw = url if isinstance(url, str) else str(url)
    u = raw if "://" in raw else "http://" + raw
    try:
        sp = urlsplit(u)
        host = sp.hostname or ""
        netloc = sp.netloc
        path = sp.path
        query = sp.query
    except Exception:
        host = netloc = path = query = ""
    return {
        "raw": raw,
        "u": u,
        "noscheme": u.split("://", 1)[-1],
        "netloc": netloc,
        "host": host,
        "path": path,
        "query": query,
    }


def _digits_ratio(s):
    return sum(c.isdigit() for c in s) / len(s) if s else 0.0


def _words(s, pat):
    return [w for w in re.split(pat, s) if w]


def _word_max(pat, key):
    def f(p):
        ws = _words(p[key], pat)
        return max((len(w) for w in ws), default=0)

    return f


def _sub_labels(host):
    parts = [l for l in host.split(".") if l]
    return parts[:-2] if len(parts) > 2 else []


def _labels(host):
    return [l for l in host.split(".") if l]


def _count(ch, key="raw"):
    def f(p):
        return p[key].count(ch)

    return f


def _hints(lst):
    def f(p):
        low = p["raw"].lower()
        return sum(low.count(h) for h in lst)

    return f


def _shortening(p):
    host = p["host"].lower()
    return int(any(s in host for s in _SHORTENERS))


def _nb_space_unq(p):
    return unquote(p["raw"]).count(" ")


def _nb_com_ci(p):
    return p["raw"].lower().count("com") - int("com" in _labels(p["host"])[-2:])


def _https_scheme(p):
    return int(p["u"].split("://", 1)[0] == "http")


def _path_ext_known(p):
    last = p["path"].rsplit("/", 1)[-1]
    if "." not in last:
        return 0
    return int(last.rsplit(".", 1)[-1].lower() in {"txt", "csv", "log"})


def _tld_in_path_host(p):
    if _IPV4.fullmatch(p["host"]):
        return 1
    labs = _labels(p["host"])
    return int(bool(labs) and labs[-1] in (p["path"] + p["query"]).lower())


def _tld_in_sub_suffix(p):
    if _IPV4.fullmatch(p["host"]):
        return 1
    parts = _labels(p["host"])
    return int(len(parts) >= 2 and parts[-1] in ".".join(parts[:-2]))


def _sus_s2(p):
    labs = [l.lower() for l in _labels(p["host"])]
    if len(labs) >= 3 and labs[-2] in _S2_LABELS:
        key = labs[-2] + "." + labs[-1]
    else:
        key = labs[-1] if labs else ""
    return int(key in _SUS_S2_KEYS)


def _abn_sub_prefix(p):
    sub = _sub_labels(p["host"])
    first = sub[0] if sub else ""
    return int(
        bool(_IPV4.fullmatch(p["host"]))
        or bool(re.match(r"\d", first))
        or bool(re.match(r"w{1,4}[-\d]", first))
    )


EXTRACTORS = {
    "length_url": [("len", lambda p: len(p["raw"]))],
    "length_hostname": [
        ("host", lambda p: len(p["host"])),
        ("netloc", lambda p: len(p["netloc"].split("@")[-1])),
    ],
    "nb_dots": [("raw", _count(".")), ("host", _count(".", "host"))],
    "nb_hyphens": [("raw", _count("-"))],
    "nb_at": [("raw", _count("@"))],
    "nb_qm": [("raw", _count("?"))],
    "nb_and": [("raw", _count("&"))],
    "nb_or": [("pipe2", _count("||")), ("pipe", _count("|"))],
    "nb_eq": [("raw", _count("="))],
    "nb_underscore": [("raw", _count("_"))],
    "nb_tilde": [("raw", _count("~"))],
    "nb_percent": [("raw", _count("%"))],
    "nb_slash": [("raw", _count("/"))],
    "nb_star": [("raw", _count("*"))],
    "nb_colon": [("raw", _count(":"))],
    "nb_comma": [("raw", _count(","))],
    "nb_semicolumn": [("raw", _count(";"))],
    "nb_dollar": [("raw", _count("$"))],
    "nb_space": [("unq", _nb_space_unq)],
    "nb_www": [("str", _count("www")), ("dot", _count("www."))],
    "nb_com": [("ci_last2", _nb_com_ci)],
    "nb_dslash": [("noscheme", _count("//", "noscheme")), ("raw", _count("//"))],
    "http_in_path": [
        ("pathquery", lambda p: (p["path"] + p["query"]).count("http")),
        ("path", lambda p: p["path"].count("http")),
    ],
    "https_token": [("scheme_http", _https_scheme)],
    "ratio_digits_url": [("r", lambda p: _digits_ratio(p["raw"]))],
    "ratio_digits_host": [("r", lambda p: _digits_ratio(p["host"]))],
    "punycode": [("xn", lambda p: int("xn--" in p["host"]))],
    "port": [
        ("re", lambda p: int(bool(_PORT.search(p["netloc"].split("@")[-1])))),
    ],
    "longest_words_raw": [
        ("pct_noscheme", _word_max(_SPLIT_PCT, "noscheme")),
    ],
    "longest_word_host": [
        ("nonalnum", _word_max(_SPLIT_NONALNUM, "host")),
        ("dot", _word_max(r"\.", "host")),
        ("urlset", _word_max(_SPLIT_URL, "host")),
    ],
    "phish_hints": [("a", _hints(_HINTS_A)), ("b", _hints(_HINTS_B))],
    "shortening_service": [("list", _shortening)],
    "path_extension": [
        ("txt_csv_log", _path_ext_known),
    ],
    "tld_in_path": [
        ("hosttld_ci", _tld_in_path_host),
    ],
    "tld_in_subdomain": [
        ("suffix_in_sub", _tld_in_sub_suffix),
    ],
    "suspecious_tld": [
        ("s2", _sus_s2),
    ],
    "abnormal_subdomain": [
        ("ip_digit_wprefix", _abn_sub_prefix),
    ],
}


def extract_features(url):
    p = _parse(url)
    out = {}
    for name, variants in EXTRACTORS.items():
        out[name] = variants[0][1](p)
    return out


def _score(cand, stored, is_float):
    a = pd.to_numeric(pd.Series(cand), errors="coerce").to_numpy(dtype=float)
    b = stored
    if is_float:
        ok = np.isclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True)
    else:
        ok = (a == b) | (np.isnan(a) & np.isnan(b))
    return float(ok.mean()), a, ok


def verify(df, write_json=True, threshold=0.99):
    missing = [n for n in EXTRACTORS if n not in df.columns]
    if missing:
        raise SystemExit(f"columns missing from dataset: {missing}")
    cand = {(name, v): [] for name, vs in EXTRACTORS.items() for v, _ in vs}
    for url in df["url"]:
        p = _parse(url)
        for name, vs in EXTRACTORS.items():
            for v, fn in vs:
                try:
                    val = fn(p)
                except Exception:
                    val = np.nan
                cand[(name, v)].append(val)

    print(f"{'feature':<24}{'variant':<14}{'match':>8}  verdict")
    print("-" * 56)
    kept = []
    dropped = []
    first_mismatch = []
    for name, vs in EXTRACTORS.items():
        stored = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        is_float = pd.api.types.is_float_dtype(df[name])
        best_v, best_s, best_a, best_ok = None, -1.0, None, None
        for v, _ in vs:
            s, a, ok = _score(cand[(name, v)], stored, is_float)
            if s > best_s:
                best_v, best_s, best_a, best_ok = v, s, a, ok
        verdict = "KEEP" if best_s >= threshold else "DROP"
        (kept if verdict == "KEEP" else dropped).append(name)
        print(f"{name:<24}{best_v:<14}{best_s * 100:>7.2f}%  {verdict}")
        if best_v != vs[0][0]:
            first_mismatch.append((name, best_v, vs[0][0]))
        if verdict == "DROP":
            for i in np.where(~best_ok)[0][:3]:
                u = str(df["url"].iloc[i])
                print(f"    e.g. {u[:88]!r} got={best_a[i]!r} want={stored[i]!r}")

    print("-" * 56)
    print(f"kept {len(kept)} / {len(EXTRACTORS)}   dropped {len(dropped)}")
    if dropped:
        print("dropped:", ", ".join(dropped))
    if first_mismatch:
        print("first-variant != best-variant:")
        for name, best_v, first_v in first_mismatch:
            print(f"  {name}: first={first_v} best={best_v}")
    if write_json:
        ordered = [c for c in df.columns if c in set(kept)]
        JSON_PATH.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
        print(f"wrote {JSON_PATH}")
    return kept, dropped


def main():
    df = load_split("train")
    print(f"train shape: {df.shape}")
    verify(df, write_json=True)


if __name__ == "__main__":
    main()
