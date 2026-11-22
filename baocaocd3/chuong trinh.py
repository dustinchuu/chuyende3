# -*- coding: utf-8 -*-
"""
DỰ ÁN: PHÂN TÍCH DỮ LIỆU MẠNG XÃ HỘI - PHÂN TÍCH TƯƠNG TÁC (LIKE, SHARE, COMMENT)

Các phần:
  1. Nạp dữ liệu (CSV thật hoặc dữ liệu mẫu)
  2. Làm sạch dữ liệu bài đăng
  3. Phân tích nội dung hiệu quả
  4. Phân tích tương tác theo thời gian
  5. Trực quan hóa engagement
  6. Dự đoán mức độ tương tác (hồi quy + phân loại)

Cách chạy:
  pip install pandas numpy matplotlib seaborn scikit-learn
  python social_media_analysis.py                 # dùng dữ liệu mẫu
  python social_media_analysis.py data/posts.csv  # dùng dữ liệu thật
"""

import os
import re
import sys
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.unicode_minus"] = False

# ======================================================================
# CẤU HÌNH
# ======================================================================
OUTPUT_DIR = "output"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
RANDOM_STATE = 42

# Ánh xạ tên cột trong file CSV của bạn -> tên cột chuẩn dùng trong code.
# Chỉ cần sửa vế TRÁI cho khớp với file thật của bạn.
COLUMN_MAP = {
    "post_id": "post_id",
    "created_time": "created_time",   # thời điểm đăng
    "post_type": "post_type",         # photo / video / text / link ...
    "caption": "caption",             # nội dung bài viết
    "likes": "likes",
    "comments": "comments",
    "shares": "shares",
    "followers": "followers",         # (tuỳ chọn) số người theo dõi lúc đăng
}

# Trọng số cho "engagement có trọng số" (share > comment > like về giá trị lan toả)
WEIGHTS = {"likes": 1, "comments": 2, "shares": 3}

STOPWORDS_VI = set("""
và của là có cho những các một được trong với này đã để không khi cũng như
từ rất đang sẽ thì mà tại về ra vào lên nhé nha ạ ơi đây đó nào gì bạn mình
""".split())


def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)


def save_fig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"   [+] Đã lưu biểu đồ: {path}")


# ======================================================================
# 1. NẠP DỮ LIỆU
# ======================================================================
def generate_sample_data(n=3000, seed=RANDOM_STATE):
    """Sinh dữ liệu mô phỏng (có cố ý chèn lỗi) để chạy thử khi chưa có dữ liệu thật."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01")
    times = start + pd.to_timedelta(rng.integers(0, 365 * 24 * 60, n), unit="m")

    types = rng.choice(["photo", "video", "text", "link", "reel"], n, p=[.30, .20, .15, .10, .25])
    followers = (20000 + (times - start).days * 25 + rng.normal(0, 300, n)).astype(int)

    topics = ["khuyến mãi", "sản phẩm mới", "mẹo hay", "review", "livestream", "tuyển dụng",
              "bí quyết", "câu chuyện", "giveaway", "thông báo", "hướng dẫn", "cảm ơn khách hàng"]
    tags_pool = ["#sale", "#new", "#tips", "#review", "#live", "#hanoi", "#viral", "#giveaway", "#trend", "#hot"]
    emojis = ["🔥", "😍", "🎉", "👉", "❤️", "✨"]

    captions, n_tags_list = [], []
    for _ in range(n):
        words = list(rng.choice(topics, rng.integers(1, 6)))
        if rng.random() < .35:
            words.append("bạn nghĩ sao?")
        if rng.random() < .30:
            words.append("https://example.com/p/" + str(rng.integers(1000, 9999)))
        k = rng.integers(0, 7)
        n_tags_list.append(k)
        words += list(rng.choice(tags_pool, k, replace=False))
        words += list(rng.choice(emojis, rng.integers(0, 4)))
        captions.append(" ".join(words))

    # Hiệu ứng ẩn: loại bài, khung giờ, cuối tuần, hashtag, câu hỏi...
    type_map = {"photo": 1.0, "video": 1.4, "text": 0.7, "link": 0.6, "reel": 1.8}
    type_eff = np.array([type_map[t] for t in types])
    hour = np.array(times.hour)
    hour_eff = 1 + 0.6 * np.exp(-((hour - 20) ** 2) / 8) + 0.3 * np.exp(-((hour - 12) ** 2) / 6)
    wk_eff = np.where(np.array(times.dayofweek) >= 5, 1.25, 1.0)
    tag_eff = 1 + 0.08 * np.minimum(n_tags_list, 5)
    q_eff = np.array([1.3 if "?" in c else 1.0 for c in captions])
    link_eff = np.array([0.8 if "http" in c else 1.0 for c in captions])
    base = followers * 0.02 * type_eff * hour_eff * wk_eff * tag_eff * q_eff * link_eff
    noise = rng.lognormal(0, 0.5, n)

    likes = rng.poisson(base * noise)
    comments = rng.poisson(base * noise * 0.10)
    shares = rng.poisson(base * noise * 0.05)

    df = pd.DataFrame({
        "post_id": np.arange(1, n + 1), "created_time": times, "post_type": types,
        "caption": captions, "likes": likes, "comments": comments, "shares": shares,
        "followers": followers,
    })

    # ---- Chèn lỗi "bẩn" để minh hoạ bước làm sạch ----
    df["created_time"] = df["created_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    idx = rng.choice(n, 60, replace=False)
    df.loc[idx, "created_time"] = pd.to_datetime(df.loc[idx, "created_time"]).dt.strftime("%d/%m/%Y %H:%M")
    df.loc[rng.choice(n, 25, replace=False), "created_time"] = "không rõ"
    df.loc[rng.choice(n, 80, replace=False), "likes"] = np.nan
    df.loc[rng.choice(n, 40, replace=False), "comments"] = np.nan
    df.loc[rng.choice(n, 30, replace=False), "shares"] = np.nan
    df.loc[rng.choice(n, 20, replace=False), "caption"] = np.nan
    df.loc[rng.choice(n, 15, replace=False), "likes"] = -5                    # giá trị âm
    df.loc[rng.choice(n, 5, replace=False), "likes"] = 9_999_999              # ngoại lệ cực đoan
    df["likes"] = df["likes"].astype("object")
    for i in rng.choice(n, 30, replace=False):                                 # dạng "1.2K"
        v = df.at[i, "likes"]
        if pd.notna(v) and isinstance(v, (int, float, np.integer)) and v > 1000:
            df.at[i, "likes"] = f"{v / 1000:.1f}K"
    df.loc[rng.choice(n, 100, replace=False), "post_type"] = \
        df.loc[rng.choice(n, 100, replace=False), "post_type"].str.upper().values[:100]
    df = pd.concat([df, df.sample(50, random_state=seed)], ignore_index=True)  # trùng lặp
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_data(path=None):
    if path and os.path.exists(path):
        print(f"[1] Đọc dữ liệu từ: {path}")
        df = pd.read_csv(path)
        rename = {v: k for k, v in COLUMN_MAP.items() if v in df.columns}
        df = df.rename(columns=rename)
    else:
        print("[1] Không có file CSV -> dùng dữ liệu mẫu (mô phỏng).")
        df = generate_sample_data()
    print(f"    Kích thước dữ liệu gốc: {df.shape}")
    return df


# ======================================================================
# 2. LÀM SẠCH DỮ LIỆU
# ======================================================================
def parse_count(x):
    """Chuyển '1.2K', '3M', '1,234', 'N/A' ... thành số."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().upper().replace(",", "")
    m = re.fullmatch(r"([\d.]+)\s*([KM]?)", s)
    if not m:
        return np.nan
    val = float(m.group(1))
    return val * {"": 1, "K": 1e3, "M": 1e6}[m.group(2)]


def parse_datetime(series):
    """Thử nhiều định dạng ngày giờ, giá trị không hợp lệ -> NaT."""
    out = pd.to_datetime(series, errors="coerce", format="%Y-%m-%d %H:%M:%S")
    mask = out.isna()
    if mask.any():
        out[mask] = pd.to_datetime(series[mask], errors="coerce", dayfirst=True)
    return out


def clean_text(s):
    s = re.sub(r"https?://\S+", " ", s)          # bỏ URL
    s = re.sub(r"[@#]\w+", " ", s)               # bỏ mention/hashtag
    s = re.sub(r"[^\w\s?!]", " ", s, flags=re.UNICODE)
    s = re.sub(r"[_\d]+", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def clean_data(df_raw):
    print("\n[2] LÀM SẠCH DỮ LIỆU")
    report = {"số dòng ban đầu": len(df_raw)}
    df = df_raw.copy()

    # 2.1 Xoá trùng lặp
    n0 = len(df)
    df = df.drop_duplicates(subset=["post_id"]) if "post_id" in df else df.drop_duplicates()
    report["trùng lặp đã xoá"] = n0 - len(df)

    # 2.2 Chuẩn hoá thời gian
    df["created_time"] = parse_datetime(df["created_time"].astype(str))
    report["thời gian không hợp lệ (bị loại)"] = int(df["created_time"].isna().sum())
    df = df.dropna(subset=["created_time"])

    # 2.3 Chuẩn hoá số liệu tương tác
    for c in ["likes", "comments", "shares"]:
        df[c] = df[c].apply(parse_count)
    neg = (df[["likes", "comments", "shares"]] < 0).sum().sum()
    report["giá trị âm -> NaN"] = int(neg)
    for c in ["likes", "comments", "shares"]:
        df.loc[df[c] < 0, c] = np.nan

    # 2.4 Điền khuyết: dùng median theo loại bài đăng
    df["post_type"] = df["post_type"].astype(str).str.strip().str.lower()
    report["giá trị thiếu (likes/comments/shares)"] = int(df[["likes", "comments", "shares"]].isna().sum().sum())
    for c in ["likes", "comments", "shares"]:
        df[c] = df[c].fillna(df.groupby("post_type")[c].transform("median")).fillna(df[c].median())

    # 2.5 Followers (nếu có)
    if "followers" in df.columns:
        df["followers"] = pd.to_numeric(df["followers"], errors="coerce")
        df["followers"] = df["followers"].fillna(df["followers"].median())
    else:
        df["followers"] = np.nan

    # 2.6 Caption
    df["caption"] = df["caption"].fillna("").astype(str)
    report["caption trống"] = int((df["caption"].str.strip() == "").sum())

    # 2.7 Xử lý ngoại lệ (winsorize theo phân vị 99.5%)
    out_cnt = 0
    for c in ["likes", "comments", "shares"]:
        cap = df[c].quantile(0.995)
        out_cnt += int((df[c] > cap).sum())
        df[c] = df[c].clip(upper=cap)
    report["ngoại lệ bị cắt ngưỡng (winsorize)"] = out_cnt

    # 2.8 Tạo đặc trưng
    df["total_engagement"] = df[["likes", "comments", "shares"]].sum(axis=1)
    df["weighted_engagement"] = sum(df[k] * w for k, w in WEIGHTS.items())
    if df["followers"].notna().any():
        df["engagement_rate"] = df["total_engagement"] / df["followers"] * 100

    # Đặc trưng thời gian
    df["hour"] = df["created_time"].dt.hour
    df["weekday"] = df["created_time"].dt.dayofweek          # 0 = Thứ Hai
    df["weekday_name"] = df["created_time"].dt.day_name()
    df["month"] = df["created_time"].dt.month
    df["date"] = df["created_time"].dt.date
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # Đặc trưng nội dung
    df["n_hashtags"] = df["caption"].str.count(r"#\w+")
    df["n_emoji"] = df["caption"].str.count(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
    df["has_link"] = df["caption"].str.contains(r"https?://", regex=True).astype(int)
    df["has_question"] = df["caption"].str.contains(r"\?").astype(int)
    df["caption_clean"] = df["caption"].apply(clean_text)
    df["caption_len"] = df["caption_clean"].str.len()
    df["word_count"] = df["caption_clean"].str.split().str.len()

    df = df.sort_values("created_time").reset_index(drop=True)
    report["số dòng sau khi làm sạch"] = len(df)

    for k, v in report.items():
        print(f"    - {k}: {v}")
    return df


# ======================================================================
# 3. PHÂN TÍCH NỘI DUNG HIỆU QUẢ
# ======================================================================
def analyze_content(df):
    print("\n[3] PHÂN TÍCH NỘI DUNG HIỆU QUẢ")
    res = {}

    # 3.1 Theo loại bài đăng
    by_type = (df.groupby("post_type")
                 .agg(so_bai=("post_id", "count"), like_tb=("likes", "mean"),
                      comment_tb=("comments", "mean"), share_tb=("shares", "mean"),
                      engagement_tb=("total_engagement", "mean"))
                 .sort_values("engagement_tb", ascending=False).round(1))
    print("\n  >> Hiệu quả theo loại bài đăng:\n", by_type)
    res["by_type"] = by_type

    # 3.2 Theo số hashtag
    df["hashtag_group"] = pd.cut(df["n_hashtags"], [-1, 0, 2, 4, 100], labels=["0", "1-2", "3-4", "5+"])
    by_tag = df.groupby("hashtag_group")["total_engagement"].agg(["count", "mean"]).round(1)
    print("\n  >> Hiệu quả theo số hashtag:\n", by_tag)
    res["by_hashtag"] = by_tag

    # 3.3 Theo độ dài caption
    df["len_group"] = pd.qcut(df["caption_len"].rank(method="first"), 4,
                              labels=["Rất ngắn", "Ngắn", "Dài", "Rất dài"])
    by_len = df.groupby("len_group")["total_engagement"].agg(["count", "mean"]).round(1)
    print("\n  >> Hiệu quả theo độ dài caption:\n", by_len)
    res["by_len"] = by_len

    # 3.4 Câu hỏi / link / emoji
    for col, label in [("has_question", "Có câu hỏi"), ("has_link", "Có link"), ("n_emoji", "Số emoji")]:
        g = df.groupby(df[col].clip(upper=3))["total_engagement"].mean().round(1)
        print(f"\n  >> Engagement trung bình theo '{label}':\n", g)

    # 3.5 Từ khoá hiệu quả: trung bình engagement của các bài chứa từ đó
    word_posts = []
    for words, eng in zip(df["caption_clean"].str.split(), df["total_engagement"]):
        for w in set(words):
            if len(w) > 2 and w not in STOPWORDS_VI:
                word_posts.append((w, eng))
    wdf = pd.DataFrame(word_posts, columns=["word", "eng"])
    kw = (wdf.groupby("word")["eng"].agg(["count", "mean"]).query("count >= 30")
             .sort_values("mean", ascending=False).round(1))
    print("\n  >> Top 10 từ khoá có engagement cao nhất:\n", kw.head(10))
    res["keywords"] = kw

    # 3.6 Hashtag hiệu quả
    tag_rows = []
    for cap, eng in zip(df["caption"], df["total_engagement"]):
        for t in set(re.findall(r"#\w+", cap.lower())):
            tag_rows.append((t, eng))
    if tag_rows:
        tdf = pd.DataFrame(tag_rows, columns=["hashtag", "eng"])
        tags = (tdf.groupby("hashtag")["eng"].agg(["count", "mean"]).query("count >= 20")
                   .sort_values("mean", ascending=False).round(1))
        print("\n  >> Hashtag hiệu quả nhất:\n", tags.head(10))
        res["hashtags"] = tags

    # 3.7 Top 10 bài tốt nhất
    top = df.nlargest(10, "total_engagement")[["created_time", "post_type", "caption", "total_engagement"]]
    res["top_posts"] = top
    return res


# ======================================================================
# 4. PHÂN TÍCH THEO THỜI GIAN
# ======================================================================
def analyze_time(df):
    print("\n[4] PHÂN TÍCH TƯƠNG TÁC THEO THỜI GIAN")
    res = {}
    day_names = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]

    res["by_hour"] = df.groupby("hour")["total_engagement"].mean().round(1)
    res["by_weekday"] = df.groupby("weekday")["total_engagement"].mean().round(1)
    res["heatmap"] = df.pivot_table(index="weekday", columns="hour",
                                    values="total_engagement", aggfunc="mean")
    res["heatmap"].index = [day_names[i] for i in res["heatmap"].index]

    monthly = df.set_index("created_time").resample("MS").agg(
        so_bai=("post_id", "count"), likes=("likes", "sum"), comments=("comments", "sum"),
        shares=("shares", "sum"), eng_tb=("total_engagement", "mean"))
    res["monthly"] = monthly

    daily = df.groupby("date")["total_engagement"].sum()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.asfreq("D").fillna(0)
    res["daily"] = daily
    res["daily_ma7"] = daily.rolling(7).mean()
    res["daily_ma30"] = daily.rolling(30).mean()

    best_hours = res["by_hour"].nlargest(3)
    best_day = res["by_weekday"].idxmax()
    print(f"    - 3 khung giờ tốt nhất: {', '.join(f'{h}h ({v:.0f})' for h, v in best_hours.items())}")
    print(f"    - Ngày trong tuần tốt nhất: {day_names[best_day]}")
    print(f"    - Tháng engagement TB cao nhất: {monthly['eng_tb'].idxmax():%m/%Y}")
    return res


# ======================================================================
# 5. TRỰC QUAN HOÁ
# ======================================================================
def visualize(df, content, timeres):
    print("\n[5] TRỰC QUAN HOÁ ENGAGEMENT")
    day_names = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]

    # 5.1 Phân phối engagement
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["total_engagement"], bins=50, kde=True, ax=ax[0], color="#4C72B0")
    ax[0].set_title("Phân phối tổng tương tác")
    sns.histplot(np.log1p(df["total_engagement"]), bins=50, kde=True, ax=ax[1], color="#55A868")
    ax[1].set_title("Phân phối log(1 + tổng tương tác)")
    save_fig("01_phan_phoi_engagement.png")

    # 5.2 Cơ cấu like/comment/share
    totals = df[["likes", "comments", "shares"]].sum()
    plt.figure(figsize=(5, 5))
    plt.pie(totals, labels=["Like", "Comment", "Share"], autopct="%1.1f%%",
            colors=["#4C72B0", "#DD8452", "#55A868"], startangle=90)
    plt.title("Cơ cấu các loại tương tác")
    save_fig("02_co_cau_tuong_tac.png")

    # 5.3 Theo loại bài đăng (cột chồng)
    bt = content["by_type"][["like_tb", "comment_tb", "share_tb"]]
    bt.plot(kind="bar", stacked=True, figsize=(8, 5), color=["#4C72B0", "#DD8452", "#55A868"])
    plt.title("Tương tác trung bình theo loại bài đăng")
    plt.ylabel("Số tương tác TB")
    plt.xticks(rotation=0)
    plt.legend(["Like", "Comment", "Share"])
    save_fig("03_theo_loai_bai.png")

    # 5.4 Boxplot theo loại
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="post_type", y="total_engagement", showfliers=False, palette="Set2")
    plt.title("Phân tán tương tác theo loại bài đăng")
    save_fig("04_boxplot_loai_bai.png")

    # 5.5 Theo giờ & thứ
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    timeres["by_hour"].plot(kind="line", marker="o", ax=ax[0], color="#C44E52")
    ax[0].set_title("Tương tác TB theo giờ đăng")
    ax[0].set_xlabel("Giờ trong ngày")
    ax[0].set_xticks(range(0, 24, 2))
    sns.barplot(x=[day_names[i] for i in timeres["by_weekday"].index],
                y=timeres["by_weekday"].values, ax=ax[1], palette="Blues_d")
    ax[1].set_title("Tương tác TB theo thứ trong tuần")
    save_fig("05_theo_gio_va_thu.png")

    # 5.6 Heatmap
    plt.figure(figsize=(13, 4.5))
    sns.heatmap(timeres["heatmap"], cmap="YlOrRd", cbar_kws={"label": "Tương tác TB"})
    plt.title("Heatmap: thứ x giờ đăng")
    plt.xlabel("Giờ")
    plt.ylabel("")
    save_fig("06_heatmap_thu_gio.png")

    # 5.7 Xu hướng theo thời gian
    plt.figure(figsize=(13, 4.5))
    plt.plot(timeres["daily"], alpha=.25, color="gray", label="Theo ngày")
    plt.plot(timeres["daily_ma7"], label="TB trượt 7 ngày", color="#4C72B0")
    plt.plot(timeres["daily_ma30"], label="TB trượt 30 ngày", color="#C44E52", lw=2)
    plt.title("Xu hướng tổng tương tác theo thời gian")
    plt.legend()
    save_fig("07_xu_huong_thoi_gian.png")

    # 5.8 Theo tháng
    m = timeres["monthly"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    m[["likes", "comments", "shares"]].plot(kind="bar", stacked=True, ax=ax,
                                            color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_xticklabels([d.strftime("%m/%Y") for d in m.index], rotation=45)
    ax.set_title("Tổng tương tác theo tháng")
    save_fig("08_theo_thang.png")

    # 5.9 Hashtag & độ dài
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    content["by_hashtag"]["mean"].plot(kind="bar", ax=ax[0], color="#8172B2", rot=0)
    ax[0].set_title("Tương tác TB theo số hashtag")
    content["by_len"]["mean"].plot(kind="bar", ax=ax[1], color="#937860", rot=0)
    ax[1].set_title("Tương tác TB theo độ dài caption")
    save_fig("09_hashtag_do_dai.png")

    # 5.10 Top từ khoá
    kw = content["keywords"].head(12).iloc[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(kw.index, kw["mean"], color="#64B5CD")
    plt.title("Top từ khoá có tương tác TB cao nhất")
    save_fig("10_top_tu_khoa.png")

    # 5.11 Tương quan
    cols = ["likes", "comments", "shares", "total_engagement", "caption_len",
            "n_hashtags", "n_emoji", "has_link", "has_question", "hour"]
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Ma trận tương quan")
    save_fig("11_tuong_quan.png")


# ======================================================================
# 6. DỰ ĐOÁN MỨC ĐỘ TƯƠNG TÁC
# ======================================================================
NUM_FEATURES = ["hour", "weekday", "month", "is_weekend", "caption_len", "word_count",
                "n_hashtags", "n_emoji", "has_link", "has_question", "followers"]
CAT_FEATURES = ["post_type"]


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])


def predict_engagement(df):
    print("\n[6] DỰ ĐOÁN MỨC ĐỘ TƯƠNG TÁC")
    feats = NUM_FEATURES + CAT_FEATURES
    data = df.copy()
    if data["followers"].isna().all():
        data["followers"] = 0

    X = data[feats]
    y = np.log1p(data["total_engagement"])        # log để giảm lệch phải

    # Chia theo thời gian: 80% bài cũ để huấn luyện, 20% bài mới để kiểm tra
    cut = int(len(data) * 0.8)
    X_tr, X_te, y_tr, y_te = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    print(f"    Train: {len(X_tr)} bài | Test: {len(X_te)} bài (chia theo thời gian)")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                               random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                       max_depth=3, random_state=RANDOM_STATE),
    }

    rows, fitted = [], {}
    for name, mdl in models.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", mdl)]).fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        pred_real, y_real = np.expm1(pred), np.expm1(y_te)
        rows.append({
            "Mô hình": name,
            "R2 (log)": r2_score(y_te, pred),
            "MAE (thực tế)": mean_absolute_error(y_real, pred_real),
            "RMSE (thực tế)": np.sqrt(mean_squared_error(y_real, pred_real)),
        })
        fitted[name] = (pipe, pred_real)
    result = pd.DataFrame(rows).set_index("Mô hình").round(3)
    print("\n  >> So sánh mô hình hồi quy:\n", result)

    best_name = result["R2 (log)"].idxmax()
    best_pipe, best_pred = fitted[best_name]
    print(f"\n    => Mô hình tốt nhất: {best_name}")

    # Biểu đồ thực tế vs dự đoán
    y_real = np.expm1(y_te)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_real, best_pred, alpha=.35, s=14, color="#4C72B0")
    lim = [0, max(y_real.max(), best_pred.max())]
    plt.plot(lim, lim, "r--", label="Dự đoán hoàn hảo")
    plt.xlabel("Thực tế")
    plt.ylabel("Dự đoán")
    plt.title(f"Thực tế vs Dự đoán ({best_name})")
    plt.legend()
    save_fig("12_thuc_te_vs_du_doan.png")

    # Độ quan trọng đặc trưng
    rf_pipe = fitted["Random Forest"][0]
    ohe_names = rf_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(CAT_FEATURES)
    names = NUM_FEATURES + list(ohe_names)
    imp = pd.Series(rf_pipe.named_steps["model"].feature_importances_, index=names).sort_values()
    plt.figure(figsize=(8, 6))
    imp.tail(15).plot(kind="barh", color="#55A868")
    plt.title("Độ quan trọng đặc trưng (Random Forest)")
    save_fig("13_feature_importance.png")
    print("\n  >> Top 5 yếu tố ảnh hưởng nhất:\n", imp.sort_values(ascending=False).head(5).round(3))

    # ---- Phân loại mức độ: Thấp / Trung bình / Cao (theo ngưỡng phân vị của tập train) ----
    q1, q2 = data["total_engagement"].iloc[:cut].quantile([1 / 3, 2 / 3])
    label = pd.cut(data["total_engagement"], [-np.inf, q1, q2, np.inf], labels=["Thấp", "Trung bình", "Cao"])
    clf = Pipeline([("prep", build_preprocessor()),
                    ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                     random_state=RANDOM_STATE, n_jobs=-1))])
    clf.fit(X_tr, label.iloc[:cut])
    pred_c = clf.predict(X_te)
    print(f"\n  >> Phân loại mức tương tác - Accuracy: {accuracy_score(label.iloc[cut:], pred_c):.3f}")
    print(classification_report(label.iloc[cut:], pred_c, zero_division=0))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(label.iloc[cut:], pred_c,
                                            labels=["Thấp", "Trung bình", "Cao"], ax=ax, cmap="Blues")
    ax.set_title("Ma trận nhầm lẫn - mức tương tác")
    save_fig("14_confusion_matrix.png")

    return best_pipe, result, clf


def demo_predict(model, df):
    """Ví dụ dự đoán cho một bài đăng mới."""
    new_post = pd.DataFrame([{
        "hour": 20, "weekday": 5, "month": 11, "is_weekend": 1, "caption_len": 80,
        "word_count": 14, "n_hashtags": 3, "n_emoji": 2, "has_link": 0, "has_question": 1,
        "followers": float(df["followers"].iloc[-1]) if df["followers"].notna().any() else 0,
        "post_type": "reel",
    }])
    pred = float(np.expm1(model.predict(new_post))[0])
    print(f"\n  >> Ví dụ: Reel đăng 20h tối thứ Bảy, có câu hỏi, 3 hashtag "
          f"-> dự đoán ~{pred:,.0f} tương tác")


# ======================================================================
# MAIN
# ======================================================================
def main():
    ensure_dirs()
    path = sys.argv[1] if len(sys.argv) > 1 else None

    df_raw = load_data(path)
    df = clean_data(df_raw)
    df.drop(columns=["date"]).to_csv(os.path.join(OUTPUT_DIR, "posts_cleaned.csv"),
                                     index=False, encoding="utf-8-sig")

    content = analyze_content(df)
    timeres = analyze_time(df)
    visualize(df, content, timeres)

    model, metrics, clf = predict_engagement(df)
    demo_predict(model, df)

    metrics.to_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"), encoding="utf-8-sig")
    content["top_posts"].to_csv(os.path.join(OUTPUT_DIR, "top_posts.csv"), index=False, encoding="utf-8-sig")
    print(f"\nHOÀN TẤT. Kết quả nằm trong thư mục '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()