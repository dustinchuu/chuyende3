# -*- coding: utf-8 -*-
"""
THU THẬP DỮ LIỆU TỪ YOUTUBE DATA API v3

Xuất ra: data/youtube_posts.csv  (nằm CÙNG THƯ MỤC với file này, không phụ thuộc Working directory)

Cách dùng:
  1. API key: đặt biến môi trường YOUTUBE_API_KEY, HOẶC tạo file api_key.txt (chỉ chứa key)
     cạnh file này. Đừng chia sẻ / đưa key lên mạng.
  2. Danh sách kênh: sửa CHANNELS bên dưới, HOẶC tạo file channels.txt (mỗi dòng một kênh:
     @handle, URL kênh, hoặc ID bắt đầu bằng UC...). Nếu có channels.txt thì dùng file đó.
  3. pip install requests pandas
     chạy file này.

Tiết kiệm quota: chỉ dùng channels.list + playlistItems.list + videos.list (mỗi lệnh 1 đơn vị,
tối đa 50 video/lần), KHÔNG dùng search.list. Nếu hết quota giữa chừng, dữ liệu đã thu được
vẫn được lưu lại.
"""

import os
import re
import sys
import time

import pandas as pd
import requests

# ======================================================================
# CẤU HÌNH
# ======================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_api_key():
    key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    key_file = os.path.join(BASE_DIR, "api_key.txt")
    if not key and os.path.exists(key_file):
        with open(key_file, encoding="utf-8") as f:
            key = f.read().strip()
    return key


API_KEY = load_api_key()
BASE_URL = "https://www.googleapis.com/youtube/v3"

# Nên chọn 5-10 kênh CÙNG LĨNH VỰC (tin tức, ẩm thực, công nghệ, giải trí...),
# mỗi kênh có từ 200 video trở lên. Thay các giá trị mẫu bên dưới bằng kênh thật.
CHANNELS = [
    "@ngoc6muitft",
    "@DTCLVN",
    "@EmChe",
    "@ghiendtcl",
    "@TrauTFT",
]


def load_channels():
    """Ưu tiên channels.txt (nếu có), nếu không dùng danh sách CHANNELS ở trên."""
    path = os.path.join(BASE_DIR, "channels.txt")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        if lines:
            return lines
    return CHANNELS

MAX_VIDEOS_PER_CHANNEL = 500          # giới hạn số video mới nhất của mỗi kênh
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "youtube_posts.csv")
LOCAL_TZ = "Asia/Ho_Chi_Minh"         # đổi giờ UTC của YouTube sang giờ Việt Nam

_quota_used = 0


# ======================================================================
# GỌI API
# ======================================================================
def api_get(endpoint, params, retries=3):
    """Gọi API, tự thử lại khi lỗi mạng; báo lỗi rõ ràng khi hết quota / key sai."""
    global _quota_used
    params = {**params, "key": API_KEY}
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
        except requests.RequestException as e:
            if attempt == retries:
                raise RuntimeError(f"Lỗi mạng khi gọi {endpoint}: {e}")
            time.sleep(2 * attempt)
            continue

        _quota_used += 1
        if r.status_code == 200:
            return r.json()

        try:
            err = r.json()["error"]["errors"][0]["reason"]
        except Exception:
            err = r.text[:200]
        if err in ("quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"):
            raise RuntimeError("Đã hết hạn mức (quota) hôm nay. Quota reset lúc 0h giờ Thái Bình Dương "
                               "(khoảng 14h-15h giờ Việt Nam). Hãy chạy lại sau.")
        if err in ("keyInvalid", "forbidden", "accessNotConfigured"):
            raise RuntimeError(f"API key không hợp lệ hoặc chưa bật YouTube Data API v3 ({err}).")
        if r.status_code >= 500 and attempt < retries:
            time.sleep(2 * attempt)
            continue
        raise RuntimeError(f"Lỗi API {r.status_code} ({endpoint}): {err}")


def resolve_channel(identifier):
    """Nhận @handle, URL kênh hoặc ID UC... và trả về thông tin kênh."""
    ident = identifier.strip()
    params = {"part": "snippet,contentDetails,statistics"}

    m = re.search(r"youtube\.com/(@[\w.\-%]+)", ident)
    if m:
        ident = m.group(1)
    m = re.search(r"youtube\.com/channel/(UC[\w\-]{22})", ident)
    if m:
        ident = m.group(1)

    if re.fullmatch(r"UC[\w\-]{22}", ident):
        params["id"] = ident
    else:
        params["forHandle"] = ident if ident.startswith("@") else "@" + ident

    items = api_get("channels", params).get("items", [])
    if not items:
        print(f"   [!] Không tìm thấy kênh: {identifier}")
        return None
    ch = items[0]
    stats = ch.get("statistics", {})
    return {
        "channel_id": ch["id"],
        "channel_title": ch["snippet"]["title"],
        "uploads_playlist": ch["contentDetails"]["relatedPlaylists"]["uploads"],
        # subscriberCount bị ẩn nếu kênh chọn ẩn số người đăng ký
        "subscribers": int(stats["subscriberCount"]) if "subscriberCount" in stats
                       and not stats.get("hiddenSubscriberCount") else None,
    }


def list_video_ids(uploads_playlist, max_videos):
    """Lấy ID các video mới nhất của kênh (playlist 'uploads')."""
    ids, token = [], None
    while len(ids) < max_videos:
        params = {"part": "contentDetails", "playlistId": uploads_playlist, "maxResults": 50}
        if token:
            params["pageToken"] = token
        data = api_get("playlistItems", params)
        ids += [it["contentDetails"]["videoId"] for it in data.get("items", [])]
        token = data.get("nextPageToken")
        if not token:
            break
    return ids[:max_videos]


def get_video_details(video_ids):
    """Lấy thông tin + thống kê video, tối đa 50 video mỗi lệnh gọi."""
    out = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        data = api_get("videos", {
            "part": "snippet,contentDetails,statistics,liveStreamingDetails",
            "id": ",".join(batch), "maxResults": 50,
        })
        out += data.get("items", [])
    return out


# ======================================================================
# XỬ LÝ DỮ LIỆU
# ======================================================================
def iso_duration_to_seconds(s):
    """'PT1H2M30S' -> 3750"""
    m = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?", s or "")
    if not m:
        return None
    d, h, mi, sec = (int(x) if x else 0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + sec


def classify_video(duration_sec, item):
    """Phân loại video theo hình thức (dùng làm 'post_type')."""
    if "liveStreamingDetails" in item:
        return "live"
    if duration_sec is None:
        return "unknown"
    if duration_sec <= 60:
        return "short"          # video ngắn (<= 60 giây)
    if duration_sec <= 600:
        return "medium"         # 1-10 phút
    return "long"               # > 10 phút


def to_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def build_rows(items, channel):
    rows = []
    for it in items:
        sn, st = it.get("snippet", {}), it.get("statistics", {})
        dur = iso_duration_to_seconds(it.get("contentDetails", {}).get("duration"))
        desc = sn.get("description", "") or ""
        hashtags = re.findall(r"#\w+", desc)[:15]
        title = sn.get("title", "") or ""
        rows.append({
            "post_id": it["id"],
            "channel_id": channel["channel_id"],
            "channel_title": channel["channel_title"],
            "created_time": (pd.to_datetime(sn["publishedAt"], utc=True)
                             .tz_convert(LOCAL_TZ).tz_localize(None)
                             .strftime("%Y-%m-%d %H:%M:%S")),
            "post_type": classify_video(dur, it),
            "caption": (title + " " + " ".join(hashtags)).strip(),
            "description": desc[:1000],
            "views": to_int(st.get("viewCount")),
            "likes": to_int(st.get("likeCount")),          # None nếu kênh ẩn lượt thích
            "comments": to_int(st.get("commentCount")),    # None nếu tắt bình luận
            "shares": None,                                # YouTube API KHÔNG cung cấp lượt chia sẻ
            "followers": channel["subscribers"],           # số đăng ký HIỆN TẠI (không phải lúc đăng)
            "duration_sec": dur,
            "category_id": sn.get("categoryId"),
            "language": sn.get("defaultAudioLanguage") or sn.get("defaultLanguage"),
        })
    return rows


def save_csv(rows):
    df = pd.DataFrame(rows).drop_duplicates(subset="post_id")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    return df


def main():
    if not API_KEY:
        sys.exit("Chưa có API key. Đặt biến môi trường YOUTUBE_API_KEY hoặc tạo file api_key.txt "
                 f"(chỉ chứa key) trong thư mục:\n  {BASE_DIR}")

    channels = load_channels()
    if any(c.startswith("@ten_kenh_") for c in channels):
        sys.exit("Danh sách kênh vẫn là giá trị mẫu (@ten_kenh_...). Hãy sửa CHANNELS trong file này "
                 "hoặc tạo file channels.txt với các kênh thật.")

    all_rows, stopped = [], None
    for ident in channels:
        print(f"\n>> Kênh: {ident}")
        try:
            ch = resolve_channel(ident)
            if not ch:
                continue
            ids = list_video_ids(ch["uploads_playlist"], MAX_VIDEOS_PER_CHANNEL)
            print(f"   {ch['channel_title']}: tìm thấy {len(ids)} video, đang lấy chi tiết...")
            rows = build_rows(get_video_details(ids), ch)
        except RuntimeError as e:
            stopped = str(e)
            print(f"   [!] Dừng ở kênh này: {e}")
            break
        all_rows += rows
        save_csv(all_rows)                      # lưu dần sau mỗi kênh để không mất dữ liệu
        print(f"   Đã lấy {len(rows)} video. (Quota đã dùng ~{_quota_used})")

    if not all_rows:
        sys.exit("Không thu được dữ liệu nào. Kiểm tra lại danh sách kênh và API key.")

    df = save_csv(all_rows)
    print(f"\nHOÀN TẤT: {len(df)} video từ {df['channel_id'].nunique()} kênh (quota ước tính ~{_quota_used} đơn vị)")
    print(f"File: {OUTPUT_PATH}")
    print(f"Ngày thu thập: {pd.Timestamp.now():%d/%m/%Y} ")
    if stopped:
        print("[!] Chưa thu thập hết các kênh (xem thông báo ở trên). Chạy lại sau khi khắc phục "
              "sẽ thu thập lại từ đầu.")


if __name__ == "__main__":
    main()