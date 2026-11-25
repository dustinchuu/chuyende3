# -*- coding: utf-8 -*-
"""Tiện ích dùng chung cho các bài kiểm thử: nạp module theo đường dẫn và dựng dữ liệu giả lập."""
import contextlib
import importlib.util
import io
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Tên file có thể khác nhau tuỳ máy (vd. bạn đổi tên thành "chuong trinh.py" hay "ytbapi.py")
ANALYSIS_FILES = ["social_media_analysis.py", "chuong trinh.py", "chuong_trinh.py", "chuongtrinh.py"]
COLLECTOR_FILES = ["youtube_collector.py", "ytbapi.py"]


def _load(alias, candidates):
    if alias in sys.modules:
        return sys.modules[alias]
    for name in candidates:
        path = os.path.join(ROOT, name)
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location(alias, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[alias] = module
            spec.loader.exec_module(module)
            return module
    raise ImportError(f"Không tìm thấy file nào trong {candidates} tại {ROOT}")


def load_analysis():
    return _load("social_media_analysis", ANALYSIS_FILES)


def load_collector():
    return _load("youtube_collector", COLLECTOR_FILES)


@contextlib.contextmanager
def quiet():
    """Tắt các dòng print của chương trình khi chạy kiểm thử."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ----------------------------------------------------------------------
# DỮ LIỆU GIẢ LẬP (định dạng giống đầu ra của youtube_collector)
# ----------------------------------------------------------------------
RAW_COLUMNS = ["post_id", "channel_id", "channel_title", "created_time", "post_type", "caption", "views",
               "likes", "comments", "shares", "followers", "duration_sec"]


def make_row(i, channel="A", rate=0.03, views=10000, post_type="medium", hour=20, days=0,
             likes=None, comments=None, caption=None, duration=300, followers=10000, created=None):
    """Một dòng dữ liệu hợp lệ; có thể ghi đè từng trường để tạo dòng lỗi."""
    if created is None:
        created = (pd.Timestamp("2024-03-01") + pd.Timedelta(days=days, hours=hour)).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "post_id": f"{channel}{i:05d}", "channel_id": f"UC_{channel}", "channel_title": f"Kenh {channel}",
        "created_time": created, "post_type": post_type,
        "caption": caption if caption is not None else f"Video số {i} #tips",
        "views": views,
        "likes": int(views * rate) if likes is None else likes,
        "comments": int(views * rate * 0.1) if comments is None else comments,
        "shares": None, "followers": followers, "duration_sec": duration,
    }


def raw_frame(rows):
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def synthetic_raw(channels, n_per, seed=0, evening_boost=1.0, short_share=None, noise=0.25):
    """Dữ liệu nhiều kênh.
    channels: danh sách (tên, tỷ lệ tương tác nền).
    evening_boost: hệ số nhân tỷ lệ khi đăng 18-23h (hiệu ứng thật, như nhau ở mọi kênh).
    short_share: danh sách xác suất video 'short' của từng kênh (để tạo yếu tố gây nhiễu theo kênh).
    Không có hiệu ứng thật nào của 'short' so với 'long' trong từng kênh."""
    rng = np.random.default_rng(seed)
    rows, k = [], 0
    for ci, (name, base) in enumerate(channels):
        p_short = 0.3 if short_share is None else short_share[ci]
        for _ in range(n_per):
            hour = int(rng.integers(0, 24))
            typ = "short" if rng.random() < p_short else "long"
            rate = base * (evening_boost if 18 <= hour <= 23 else 1.0) * rng.lognormal(0, noise)
            views = int(rng.integers(5000, 60000))
            q = "?" if rng.random() < 0.3 else ""
            rows.append(make_row(k, channel=name, rate=rate, views=views, post_type=typ, hour=hour,
                                 days=int(rng.integers(0, 900)), duration=40 if typ == "short" else 1200,
                                 caption=(f"Tiêu đề {rng.choice(['học', 'review', 'mẹo hay', 'vlog', 'tin tức mới'])} "
                                          f"{k} {q} {'#tips' if rng.random() < 0.5 else ''}").strip(),
                                 followers=10000 * (ci + 1)))
            k += 1
    return raw_frame(rows)