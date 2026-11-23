# -*- coding: utf-8 -*-
"""
DỰ ÁN: PHÂN TÍCH DỮ LIỆU MẠNG XÃ HỘI - PHÂN TÍCH TƯƠNG TÁC (LIKE, COMMENT) TRÊN YOUTUBE

Dữ liệu: file data/youtube_posts.csv do youtube_collector.py (YouTube Data API v3) tạo ra,
nằm CÙNG THƯ MỤC với file này -> không phụ thuộc vào Working directory của PyCharm.

Các phần:
  1. Nạp dữ liệu
  2. Làm sạch dữ liệu
  3. Phân tích nội dung hiệu quả
  4. Phân tích tương tác theo thời gian
  5. Trực quan hóa engagement
  6. Dự đoán mức độ tương tác (hồi quy + phân loại)

Cách chạy:
  pip install pandas numpy matplotlib seaborn scikit-learn
  python <tên file này>.py                    # tự đọc data/youtube_posts.csv cạnh file
  python <tên file này>.py duong/dan/khac.csv # (tuỳ chọn) dùng file CSV khác

Lưu ý: YouTube API không có lượt chia sẻ (share) -> cột share bị bỏ qua. Thước đo chính là
"tỷ lệ tương tác trên lượt xem" = (like + comment) / view, để loại ảnh hưởng của tuổi video.
"""

import os
import re
import sys
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")   # chỉ lưu ảnh, không mở cửa sổ -> tránh lỗi tkinter khi chạy trong PyCharm
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))     # thư mục chứa file này
DATA_PATH = os.path.join(BASE_DIR, "data", "youtube_posts.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
RANDOM_STATE = 42

# Ánh xạ tên cột trong file CSV của bạn -> tên cột chuẩn dùng trong code.
# Chỉ cần sửa vế TRÁI cho khớp với file thật của bạn.
COLUMN_MAP = {
    "post_id": "post_id",
    "created_time": "created_time",   # thời điểm đăng
    "post_type": "post_type",         # short / medium / long / live
    "caption": "caption",             # tiêu đề video + hashtag
    "likes": "likes",
    "comments": "comments",
    "shares": "shares",
    "followers": "followers",         # (tuỳ chọn) số người theo dõi / đăng ký
    "views": "views",                 # (tuỳ chọn) lượt xem - có với YouTube
    "duration_sec": "duration_sec",   # (tuỳ chọn) độ dài video (giây)
}

# Chỉ giữ bài đã đăng ít nhất N ngày, vì số liệu bài quá mới chưa ổn định (có thể đặt 0 để tắt)
MIN_AGE_DAYS = 30

# Thước đo chính của toàn bộ phân tích (xem docstring)
METRIC_LABEL = "Tỷ lệ tương tác/lượt xem (%)"

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
REQUIRED_COLUMNS = ["post_id", "created_time", "post_type", "caption", "likes", "comments", "views"]


def load_data(path):
    if not os.path.exists(path):
        sys.exit(f"Không tìm thấy file dữ liệu:\n  {path}\n"
                 "-> Hãy chạy file thu thập (youtube_collector) trước để tạo data/youtube_posts.csv.")
    print(f"[1] Đọc dữ liệu từ: {path}")
    df = pd.read_csv(path)
    df = df.rename(columns={v: k for k, v in COLUMN_MAP.items() if v in df.columns})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"File CSV thiếu cột bắt buộc: {missing}\nCác cột hiện có: {list(df.columns)}")
    print(f"    Kích thước dữ liệu gốc: {df.shape}")
    if "channel_title" in df.columns:
        print("    Số video theo kênh:")
        for k, v in df["channel_title"].value_counts().items():
            print(f"      - {k}: {v}")
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
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"[_\d]+", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def clean_data(df_raw):
    print("\n[2] LÀM SẠCH DỮ LIỆU")
    report = {"số dòng ban đầu": len(df_raw)}
    df = df_raw.copy()

    # 2.1 Xoá trùng lặp (cùng một video)
    n0 = len(df)
    df = df.drop_duplicates(subset=["post_id"])
    report["video trùng lặp đã xoá"] = n0 - len(df)

    # 2.2 Chuẩn hoá thời gian
    df["created_time"] = parse_datetime(df["created_time"].astype(str))
    n0 = len(df)
    df = df.dropna(subset=["created_time"])
    report["thời gian không hợp lệ (bị loại)"] = n0 - len(df)

    # 2.3 Loại video quá mới (số liệu chưa ổn định)
    if MIN_AGE_DAYS > 0:
        n0 = len(df)
        df = df[df["created_time"] <= pd.Timestamp.now() - pd.Timedelta(days=MIN_AGE_DAYS)]
        report[f"video đăng chưa đủ {MIN_AGE_DAYS} ngày (bị loại)"] = n0 - len(df)

    # 2.4 Chuẩn hoá số liệu tương tác
    for c in ["likes", "comments", "views"]:
        df[c] = df[c].apply(parse_count)
    df["shares"] = 0                    # YouTube API không cung cấp lượt chia sẻ
    report["cột shares"] = "YouTube không cung cấp -> bỏ qua trong phân tích"

    n0 = len(df)
    df = df[df["likes"].notna() & (df["views"] > 0)]            # kênh ẩn lượt thích / video chưa có lượt xem
    report["video ẩn lượt thích hoặc 0 lượt xem (bị loại)"] = n0 - len(df)

    report["giá trị âm -> NaN"] = int((df[["likes", "comments"]] < 0).sum().sum())
    for c in ["likes", "comments"]:
        df.loc[df[c] < 0, c] = np.nan
    report["video tắt bình luận (comment trống -> 0)"] = int(df["comments"].isna().sum())
    df["comments"] = df["comments"].fillna(0)

    # likes không được vượt views (dữ liệu bất thường)
    n0 = len(df)
    df = df[df["likes"] <= df["views"]]
    report["video có like > view (bị loại)"] = n0 - len(df)

    # 2.5 Thời lượng, người đăng ký, loại video
    if "duration_sec" in df.columns:
        df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce")
        n0 = len(df)
        df = df[df["duration_sec"].isna() | (df["duration_sec"] > 0)]   # video 0 giây (premiere/live chưa phát)
        report["video thời lượng 0 giây (bị loại)"] = n0 - len(df)
        df["duration_sec"] = df["duration_sec"].fillna(df["duration_sec"].median())
    if "followers" in df.columns:
        df["followers"] = pd.to_numeric(df["followers"], errors="coerce")
        report["video thiếu số đăng ký kênh (điền median)"] = int(df["followers"].isna().sum())
        df["followers"] = df["followers"].fillna(df["followers"].median())
    else:
        df["followers"] = np.nan
    df["post_type"] = df["post_type"].astype(str).str.strip().str.lower()

    # 2.6 Tiêu đề / caption
    df["caption"] = df["caption"].fillna("").astype(str)
    report["tiêu đề trống"] = int((df["caption"].str.strip() == "").sum())

    # 2.7 Xử lý ngoại lệ (winsorize theo phân vị 99.5%)
    out_cnt = 0
    for c in ["likes", "comments"]:
        cap = df[c].quantile(0.995)
        out_cnt += int((df[c] > cap).sum())
        df[c] = df[c].clip(upper=cap)
    report["ngoại lệ bị cắt ngưỡng (winsorize)"] = out_cnt

    # 2.8 Tạo đặc trưng
    df["total_engagement"] = df[["likes", "comments", "shares"]].sum(axis=1)
    df["weighted_engagement"] = sum(df[k] * w for k, w in WEIGHTS.items())
    df["engagement_rate"] = df["total_engagement"] / df["views"] * 100
    df["target_metric"] = df["engagement_rate"]
    print(f"    - Thước đo chính được dùng: {METRIC_LABEL}")

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
    if len(df) < 200:
        print("    [!] CẢNH BÁO: còn dưới 200 video, kết quả dự đoán sẽ kém tin cậy. "
              "Hãy thu thập thêm kênh/video.")
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
                      engagement_tb=("target_metric", "mean"),
                      engagement_median=("target_metric", "median"))
                 .sort_values("engagement_tb", ascending=False).round(1))
    print("\n  >> Hiệu quả theo loại bài đăng:\n", by_type)
    res["by_type"] = by_type

    # 3.2 Theo số hashtag
    df["hashtag_group"] = pd.cut(df["n_hashtags"], [-1, 0, 2, 4, 100], labels=["0", "1-2", "3-4", "5+"])
    by_tag = df.groupby("hashtag_group")["target_metric"].agg(["count", "mean"]).round(1)
    print("\n  >> Hiệu quả theo số hashtag:\n", by_tag)
    res["by_hashtag"] = by_tag

    # 3.3 Theo độ dài caption
    df["len_group"] = pd.qcut(df["caption_len"].rank(method="first"), 4,
                              labels=["Rất ngắn", "Ngắn", "Dài", "Rất dài"])
    by_len = df.groupby("len_group")["target_metric"].agg(["count", "mean"]).round(1)
    print("\n  >> Hiệu quả theo độ dài caption:\n", by_len)
    res["by_len"] = by_len

    # 3.4 Câu hỏi / link / emoji
    for col, label in [("has_question", "Có câu hỏi"), ("has_link", "Có link"), ("n_emoji", "Số emoji")]:
        g = df.groupby(df[col].clip(upper=3))["target_metric"].mean().round(1)
        print(f"\n  >> {METRIC_LABEL} trung bình theo '{label}':\n", g)

    # 3.5 Từ khoá hiệu quả: trung bình engagement của các bài chứa từ đó
    word_posts = []
    for words, eng in zip(df["caption_clean"].str.split(), df["target_metric"]):
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
    for cap, eng in zip(df["caption"], df["target_metric"]):
        for t in set(re.findall(r"#\w+", cap.lower())):
            tag_rows.append((t, eng))
    if tag_rows:
        tdf = pd.DataFrame(tag_rows, columns=["hashtag", "eng"])
        tags = (tdf.groupby("hashtag")["eng"].agg(["count", "mean"]).query("count >= 20")
                   .sort_values("mean", ascending=False).round(1))
        print("\n  >> Hashtag hiệu quả nhất:\n", tags.head(10))
        res["hashtags"] = tags

    # 3.7 Top 10 bài tốt nhất
    top = df.nlargest(10, "target_metric")[["post_id", "created_time", "post_type", "caption",
                                             "total_engagement", "target_metric"]]
    res["top_posts"] = top
    return res


# ======================================================================
# 4. PHÂN TÍCH THEO THỜI GIAN
# ======================================================================
def analyze_time(df):
    print("\n[4] PHÂN TÍCH TƯƠNG TÁC THEO THỜI GIAN")
    res = {}
    day_names = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]

    res["by_hour"] = df.groupby("hour")["target_metric"].mean().round(1)
    res["by_weekday"] = df.groupby("weekday")["target_metric"].mean().round(1)
    res["heatmap"] = df.pivot_table(index="weekday", columns="hour",
                                    values="target_metric", aggfunc="mean")
    res["heatmap"].index = [day_names[i] for i in res["heatmap"].index]

    monthly = df.set_index("created_time").resample("MS").agg(
        so_bai=("post_id", "count"), likes=("likes", "sum"), comments=("comments", "sum"),
        shares=("shares", "sum"), eng_tb=("target_metric", "mean"))
    res["monthly"] = monthly

    weekly = df.set_index("created_time")["target_metric"].resample("W").mean()
    res["daily"] = weekly                                   # (giữ tên khoá cũ) trung bình theo tuần
    res["daily_ma7"] = weekly.rolling(4, min_periods=1).mean()
    res["daily_ma30"] = weekly.rolling(12, min_periods=1).mean()

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
    sns.histplot(df["target_metric"], bins=50, kde=True, ax=ax[0], color="#4C72B0")
    ax[0].set_title(f"Phân phối: {METRIC_LABEL}")
    sns.histplot(np.log1p(df["target_metric"]), bins=50, kde=True, ax=ax[1], color="#55A868")
    ax[1].set_title("Phân phối log(1 + giá trị)")
    save_fig("01_phan_phoi_engagement.png")

    # 5.2 Cơ cấu like/comment/share
    totals = df[["likes", "comments", "shares"]].sum()
    totals = totals[totals > 0]
    names_vi = {"likes": "Like", "comments": "Comment", "shares": "Share"}
    plt.figure(figsize=(5, 5))
    plt.pie(totals, labels=[names_vi[i] for i in totals.index], autopct="%1.1f%%",
            colors=["#4C72B0", "#DD8452", "#55A868"][:len(totals)], startangle=90)
    plt.title("Cơ cấu các loại tương tác")
    save_fig("02_co_cau_tuong_tac.png")

    # 5.3 Theo loại bài đăng (cột chồng)
    bt = content["by_type"][["like_tb", "comment_tb", "share_tb"]]
    bt = bt.loc[:, bt.sum() > 0]
    bt.plot(kind="bar", stacked=True, figsize=(8, 5), color=["#4C72B0", "#DD8452", "#55A868"][:bt.shape[1]])
    plt.title("Số like/comment/share trung bình theo loại bài đăng")
    plt.ylabel("Số lượng TB")
    plt.xticks(rotation=0)
    plt.legend([{"like_tb": "Like", "comment_tb": "Comment", "share_tb": "Share"}[c] for c in bt.columns])
    save_fig("03_theo_loai_bai.png")

    # 5.4 Boxplot theo loại
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="post_type", y="target_metric", showfliers=False, palette="Set2")
    plt.title(f"Phân tán {METRIC_LABEL} theo loại bài đăng")
    plt.ylabel(METRIC_LABEL)
    save_fig("04_boxplot_loai_bai.png")

    # 5.5 Theo giờ & thứ
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    timeres["by_hour"].plot(kind="line", marker="o", ax=ax[0], color="#C44E52")
    ax[0].set_title(f"{METRIC_LABEL} TB theo giờ đăng")
    ax[0].set_xlabel("Giờ trong ngày")
    ax[0].set_xticks(range(0, 24, 2))
    sns.barplot(x=[day_names[i] for i in timeres["by_weekday"].index],
                y=timeres["by_weekday"].values, ax=ax[1], palette="Blues_d")
    ax[1].set_title(f"{METRIC_LABEL} TB theo thứ")
    save_fig("05_theo_gio_va_thu.png")

    # 5.6 Heatmap
    plt.figure(figsize=(13, 4.5))
    sns.heatmap(timeres["heatmap"], cmap="YlOrRd", cbar_kws={"label": f"{METRIC_LABEL} TB"})
    plt.title("Heatmap: thứ x giờ đăng")
    plt.xlabel("Giờ")
    plt.ylabel("")
    save_fig("06_heatmap_thu_gio.png")

    # 5.7 Xu hướng theo thời gian
    plt.figure(figsize=(13, 4.5))
    plt.plot(timeres["daily"], alpha=.3, color="gray", label="TB theo tuần")
    plt.plot(timeres["daily_ma7"], label="TB trượt 4 tuần", color="#4C72B0")
    plt.plot(timeres["daily_ma30"], label="TB trượt 12 tuần", color="#C44E52", lw=2)
    plt.title(f"Xu hướng {METRIC_LABEL} theo thời gian")
    plt.legend()
    save_fig("07_xu_huong_thoi_gian.png")

    # 5.8 Theo tháng
    m = timeres["monthly"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    mm = m[["likes", "comments", "shares"]]
    mm = mm.loc[:, mm.sum() > 0]
    mm.plot(kind="bar", stacked=True, ax=ax, color=["#4C72B0", "#DD8452", "#55A868"][:mm.shape[1]])
    ax.set_xticklabels([d.strftime("%m/%Y") for d in m.index], rotation=45)
    ax.set_title("Tổng like/comment/share của các bài đăng theo tháng")
    save_fig("08_theo_thang.png")

    # 5.9 Hashtag & độ dài
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    content["by_hashtag"]["mean"].plot(kind="bar", ax=ax[0], color="#8172B2", rot=0)
    ax[0].set_title(f"{METRIC_LABEL} TB theo số hashtag")
    content["by_len"]["mean"].plot(kind="bar", ax=ax[1], color="#937860", rot=0)
    ax[1].set_title(f"{METRIC_LABEL} TB theo độ dài caption")
    save_fig("09_hashtag_do_dai.png")

    # 5.10 Top từ khoá
    kw = content["keywords"].head(12).iloc[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(kw.index, kw["mean"], color="#64B5CD")
    plt.title(f"Top từ khoá có {METRIC_LABEL} TB cao nhất")
    save_fig("10_top_tu_khoa.png")

    # 5.11 Tương quan
    cols = ["likes", "comments", "shares", "total_engagement", "target_metric", "caption_len",
            "n_hashtags", "n_emoji", "has_link", "has_question", "hour"]
    if "views" in df.columns:
        cols.insert(0, "views")
    if "duration_sec" in df.columns:
        cols.append("duration_sec")
    cols = [c for c in cols if df[c].nunique() > 1]
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Ma trận tương quan")
    save_fig("11_tuong_quan.png")


# ======================================================================
# 6. DỰ ĐOÁN MỨC ĐỘ TƯƠNG TÁC
# ======================================================================
# Chỉ dùng các đặc trưng BIẾT TRƯỚC khi đăng bài (không dùng views/likes để tránh rò rỉ dữ liệu)
CANDIDATE_NUM = ["hour", "weekday", "month", "is_weekend", "caption_len", "word_count",
                 "n_hashtags", "n_emoji", "has_link", "has_question", "followers", "duration_sec"]
CAT_FEATURES = ["post_type"]


def build_preprocessor(num_features):
    return ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])


def predict_engagement(df):
    print("\n[6] DỰ ĐOÁN MỨC ĐỘ TƯƠNG TÁC")
    data = df.copy()
    if data["followers"].isna().all():
        data["followers"] = 0
    # bỏ đặc trưng không có hoặc không đổi (vd. duration_sec khi dữ liệu không có)
    num_features = [c for c in CANDIDATE_NUM if c in data.columns and data[c].nunique() > 1]
    feats = num_features + CAT_FEATURES
    print(f"    Thước đo dự đoán: {METRIC_LABEL}")
    print(f"    Đặc trưng số: {num_features}")

    X = data[feats]
    y = np.log1p(data["target_metric"])        # log để giảm lệch phải

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
        pipe = Pipeline([("prep", build_preprocessor(num_features)), ("model", mdl)]).fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        pred_real, y_real = np.expm1(pred), np.expm1(y_te)
        rows.append({
            "Mô hình": name,
            "R2 (log)": r2_score(y_te, pred),
            "MAE (đơn vị thước đo)": mean_absolute_error(y_real, pred_real),
            "RMSE (đơn vị thước đo)": np.sqrt(mean_squared_error(y_real, pred_real)),
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
    names = num_features + list(ohe_names)
    imp = pd.Series(rf_pipe.named_steps["model"].feature_importances_, index=names).sort_values()
    plt.figure(figsize=(8, 6))
    imp.tail(15).plot(kind="barh", color="#55A868")
    plt.title("Độ quan trọng đặc trưng (Random Forest)")
    save_fig("13_feature_importance.png")
    print("\n  >> Top 5 yếu tố ảnh hưởng nhất:\n", imp.sort_values(ascending=False).head(5).round(3))

    # ---- Phân loại mức độ: Thấp / Trung bình / Cao (theo ngưỡng phân vị của tập train) ----
    q1, q2 = data["target_metric"].iloc[:cut].quantile([1 / 3, 2 / 3])
    label = pd.cut(data["target_metric"], [-np.inf, q1, q2, np.inf], labels=["Thấp", "Trung bình", "Cao"])
    clf = Pipeline([("prep", build_preprocessor(num_features)),
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

    return best_pipe, result, clf, num_features


def demo_predict(model, df, num_features):
    """Ví dụ dự đoán cho một bài đăng mới (loại bài phổ biến nhất, 20h tối thứ Bảy)."""
    post_type = df["post_type"].mode().iloc[0]
    row = {"hour": 20, "weekday": 5, "month": 11, "is_weekend": 1, "caption_len": 80,
           "word_count": 14, "n_hashtags": 3, "n_emoji": 2, "has_link": 0, "has_question": 1,
           "followers": float(df["followers"].median()) if df["followers"].notna().any() else 0,
           "duration_sec": float(df["duration_sec"].median()) if "duration_sec" in df else 0,
           "post_type": post_type}
    new_post = pd.DataFrame([row])[num_features + CAT_FEATURES]
    pred = float(np.expm1(model.predict(new_post))[0])
    print(f"\n  >> Ví dụ: bài loại '{post_type}' đăng 20h tối thứ Bảy, có câu hỏi, 3 hashtag "
          f"-> dự đoán {METRIC_LABEL} ~{pred:,.2f}")


# ======================================================================
# MAIN
# ======================================================================
def main():
    ensure_dirs()
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH

    df_raw = load_data(path)
    df = clean_data(df_raw)
    df.drop(columns=["date"]).to_csv(os.path.join(OUTPUT_DIR, "posts_cleaned.csv"),
                                     index=False, encoding="utf-8-sig")

    content = analyze_content(df)
    timeres = analyze_time(df)
    visualize(df, content, timeres)

    model, metrics, clf, num_features = predict_engagement(df)
    demo_predict(model, df, num_features)

    metrics.to_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"), encoding="utf-8-sig")
    content["top_posts"].to_csv(os.path.join(OUTPUT_DIR, "top_posts.csv"), index=False, encoding="utf-8-sig")
    print(f"\nHOÀN TẤT. Kết quả nằm trong thư mục:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()