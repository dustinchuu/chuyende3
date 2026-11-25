# -*- coding: utf-8 -*-
"""Kiểm thử file thu thập dữ liệu YouTube (không cần mạng, không dùng API key thật)."""
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

from helpers import load_collector, quiet

yc = load_collector()


class FakeResponse:
    def __init__(self, payload=None, status=200):
        self._payload, self.status_code, self.text = payload or {}, status, str(payload)

    def json(self):
        return self._payload


def error_payload(reason):
    return {"error": {"errors": [{"reason": reason}]}}


class TestDuration(unittest.TestCase):
    def test_common_formats(self):
        self.assertEqual(yc.iso_duration_to_seconds("PT45S"), 45)
        self.assertEqual(yc.iso_duration_to_seconds("PT5M10S"), 310)
        self.assertEqual(yc.iso_duration_to_seconds("PT1H2M3S"), 3723)
        self.assertEqual(yc.iso_duration_to_seconds("PT2H"), 7200)

    def test_days_and_zero(self):
        self.assertEqual(yc.iso_duration_to_seconds("P1DT1H"), 90000)
        self.assertEqual(yc.iso_duration_to_seconds("P0D"), 0)       # video chưa phát / premiere

    def test_invalid_returns_none(self):
        self.assertIsNone(yc.iso_duration_to_seconds("abc"))
        self.assertIsNone(yc.iso_duration_to_seconds(None))
        self.assertIsNone(yc.iso_duration_to_seconds(""))


class TestClassifyVideo(unittest.TestCase):
    def test_boundaries(self):
        c = lambda d, item=None: yc.classify_video(d, item or {})
        self.assertEqual(c(1), "short")
        self.assertEqual(c(60), "short")
        self.assertEqual(c(61), "medium")
        self.assertEqual(c(600), "medium")
        self.assertEqual(c(601), "long")

    def test_live_and_unknown(self):
        self.assertEqual(yc.classify_video(3000, {"liveStreamingDetails": {}}), "live")
        self.assertEqual(yc.classify_video(None, {}), "unknown")


class TestToInt(unittest.TestCase):
    def test_values(self):
        self.assertEqual(yc.to_int("123"), 123)
        self.assertEqual(yc.to_int(7), 7)
        self.assertIsNone(yc.to_int(None))
        self.assertIsNone(yc.to_int("abc"))


class TestBuildRows(unittest.TestCase):
    def setUp(self):
        self.channel = {"channel_id": "UC1", "channel_title": "Kenh Test", "subscribers": 5000}
        self.item = {
            "id": "vid1",
            "snippet": {"title": "Tiêu đề hay?", "description": "mô tả #mot #hai https://x.com",
                        "publishedAt": "2025-01-01T17:00:00Z", "categoryId": "22"},
            "contentDetails": {"duration": "PT45S"},
            "statistics": {"viewCount": "1000", "likeCount": "50", "commentCount": "5"},
        }

    def test_basic_fields(self):
        row = yc.build_rows([self.item], self.channel)[0]
        self.assertEqual(row["post_id"], "vid1")
        self.assertEqual(row["views"], 1000)
        self.assertEqual(row["likes"], 50)
        self.assertEqual(row["comments"], 5)
        self.assertEqual(row["post_type"], "short")
        self.assertEqual(row["followers"], 5000)
        self.assertEqual(row["duration_sec"], 45)

    def test_timezone_converted_to_vietnam_time(self):
        row = yc.build_rows([self.item], self.channel)[0]
        self.assertEqual(row["created_time"], "2025-01-02 00:00:00")       # UTC+7

    def test_caption_has_title_and_hashtags_only(self):
        row = yc.build_rows([self.item], self.channel)[0]
        self.assertIn("Tiêu đề hay?", row["caption"])
        self.assertIn("#mot", row["caption"])
        self.assertNotIn("https", row["caption"])

    def test_shares_is_always_none(self):
        self.assertIsNone(yc.build_rows([self.item], self.channel)[0]["shares"])

    def test_hidden_likes_and_disabled_comments_become_none(self):
        self.item["statistics"] = {"viewCount": "10"}
        row = yc.build_rows([self.item], self.channel)[0]
        self.assertIsNone(row["likes"])
        self.assertIsNone(row["comments"])


class TestKeyAndChannelFiles(unittest.TestCase):
    def test_key_from_env(self):
        with mock.patch.dict(os.environ, {"YOUTUBE_API_KEY": "  ENVKEY  "}):
            self.assertEqual(yc.load_api_key(), "ENVKEY")

    def test_key_from_file_when_env_missing(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "api_key.txt"), "w", encoding="utf-8") as f:
                f.write("FILEKEY\n")
            env = {k: v for k, v in os.environ.items() if k != "YOUTUBE_API_KEY"}
            with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(yc, "BASE_DIR", d):
                self.assertEqual(yc.load_api_key(), "FILEKEY")

    def test_no_key_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            env = {k: v for k, v in os.environ.items() if k != "YOUTUBE_API_KEY"}
            with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(yc, "BASE_DIR", d):
                self.assertEqual(yc.load_api_key(), "")

    def test_channels_file_ignores_comments_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "channels.txt"), "w", encoding="utf-8") as f:
                f.write("# ghi chú\n@kenh1\n\n  @kenh2  \n")
            with mock.patch.object(yc, "BASE_DIR", d):
                self.assertEqual(yc.load_channels(), ["@kenh1", "@kenh2"])

    def test_channels_fallback_to_constant(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(yc, "BASE_DIR", d):
            self.assertEqual(yc.load_channels(), yc.CHANNELS)


class TestApiGet(unittest.TestCase):
    def test_success(self):
        with mock.patch.object(yc.requests, "get", return_value=FakeResponse({"ok": 1})):
            self.assertEqual(yc.api_get("videos", {}), {"ok": 1})

    def test_quota_exceeded_raises_clear_error(self):
        with mock.patch.object(yc.requests, "get", return_value=FakeResponse(error_payload("quotaExceeded"), 403)):
            with self.assertRaises(RuntimeError) as cm:
                yc.api_get("videos", {})
            self.assertIn("quota", str(cm.exception).lower())

    def test_invalid_key_raises_clear_error(self):
        with mock.patch.object(yc.requests, "get", return_value=FakeResponse(error_payload("keyInvalid"), 400)):
            with self.assertRaises(RuntimeError) as cm:
                yc.api_get("videos", {})
            self.assertIn("key", str(cm.exception).lower())

    def test_retries_after_server_error(self):
        responses = [FakeResponse(error_payload("backendError"), 500), FakeResponse({"ok": 2})]
        with mock.patch.object(yc.requests, "get", side_effect=responses), mock.patch.object(yc.time, "sleep"):
            self.assertEqual(yc.api_get("videos", {}), {"ok": 2})

    def test_network_error_gives_up_after_retries(self):
        with mock.patch.object(yc.requests, "get", side_effect=yc.requests.ConnectionError("boom")), \
                mock.patch.object(yc.time, "sleep"):
            with self.assertRaises(RuntimeError):
                yc.api_get("videos", {}, retries=2)

    def test_api_key_sent_with_request(self):
        with mock.patch.object(yc, "API_KEY", "SECRET"), \
                mock.patch.object(yc.requests, "get", return_value=FakeResponse({})) as g:
            yc.api_get("videos", {"part": "snippet"})
            self.assertEqual(g.call_args.kwargs["params"]["key"], "SECRET")


class TestResolveChannel(unittest.TestCase):
    @staticmethod
    def payload(subs="1000", hidden=False):
        stats = {"hiddenSubscriberCount": True} if hidden else {"subscriberCount": subs}
        return {"items": [{"id": "UCabc", "snippet": {"title": "T"},
                           "contentDetails": {"relatedPlaylists": {"uploads": "UUabc"}}, "statistics": stats}]}

    def call(self, ident, payload=None):
        with mock.patch.object(yc, "api_get", return_value=payload or self.payload()) as g:
            return yc.resolve_channel(ident), g.call_args.args[1]

    def test_handle(self):
        info, params = self.call("@abc")
        self.assertEqual(params["forHandle"], "@abc")
        self.assertEqual(info["subscribers"], 1000)
        self.assertEqual(info["uploads_playlist"], "UUabc")

    def test_handle_without_at_sign(self):
        _, params = self.call("abc")
        self.assertEqual(params["forHandle"], "@abc")

    def test_channel_id(self):
        _, params = self.call("UC" + "a" * 22)
        self.assertEqual(params["id"], "UC" + "a" * 22)

    def test_url_with_handle(self):
        _, params = self.call("https://www.youtube.com/@abc")
        self.assertEqual(params["forHandle"], "@abc")

    def test_url_with_channel_id(self):
        _, params = self.call("https://www.youtube.com/channel/UC" + "b" * 22)
        self.assertEqual(params["id"], "UC" + "b" * 22)

    def test_hidden_subscribers_gives_none(self):
        info, _ = self.call("@abc", self.payload(hidden=True))
        self.assertIsNone(info["subscribers"])

    def test_not_found_returns_none(self):
        with quiet(), mock.patch.object(yc, "api_get", return_value={"items": []}):
            self.assertIsNone(yc.resolve_channel("@khongco"))


class TestPagingAndBatching(unittest.TestCase):
    def test_list_video_ids_follows_pages_and_respects_limit(self):
        pages = {None: ("p2", 50), "p2": ("p3", 50), "p3": (None, 50)}

        def fake(endpoint, params):
            nxt, n = pages[params.get("pageToken")]
            start = {None: 0, "p2": 50, "p3": 100}[params.get("pageToken")]
            d = {"items": [{"contentDetails": {"videoId": f"v{start + i}"}} for i in range(n)]}
            if nxt:
                d["nextPageToken"] = nxt
            return d

        with mock.patch.object(yc, "api_get", side_effect=fake):
            self.assertEqual(len(yc.list_video_ids("UU", 500)), 150)
            ids = yc.list_video_ids("UU", 75)
        self.assertEqual(len(ids), 75)
        self.assertEqual(ids[0], "v0")

    def test_get_video_details_batches_of_50(self):
        calls = []

        def fake(endpoint, params):
            ids = params["id"].split(",")
            calls.append(len(ids))
            return {"items": [{"id": i} for i in ids]}

        with mock.patch.object(yc, "api_get", side_effect=fake):
            items = yc.get_video_details([f"v{i}" for i in range(120)])
        self.assertEqual(calls, [50, 50, 20])
        self.assertEqual(len(items), 120)


class TestMain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, "data", "youtube_posts.csv")

    def tearDown(self):
        self.tmp.cleanup()

    def fake_item(self, vid):
        return {"id": vid,
                "snippet": {"title": f"T {vid}", "description": "", "publishedAt": "2025-02-01T10:00:00Z"},
                "contentDetails": {"duration": "PT5M"},
                "statistics": {"viewCount": "100", "likeCount": "5", "commentCount": "1"}}

    def test_exits_without_api_key(self):
        with quiet(), mock.patch.object(yc, "API_KEY", ""):
            with self.assertRaises(SystemExit):
                yc.main()

    def test_exits_when_channels_are_placeholders(self):
        with quiet(), mock.patch.object(yc, "API_KEY", "K"), \
                mock.patch.object(yc, "load_channels", return_value=["@ten_kenh_1"]):
            with self.assertRaises(SystemExit):
                yc.main()

    def test_happy_path_writes_csv(self):
        info = lambda ident: {"channel_id": "UC" + ident, "channel_title": ident, "uploads_playlist": "UU" + ident,
                              "subscribers": 10}
        # mã video phải khác nhau giữa các kênh (như thực tế), vì dữ liệu được loại trùng theo post_id
        listing = lambda playlist, limit: [f"{playlist}_v1", f"{playlist}_v2"]
        with quiet(), mock.patch.object(yc, "API_KEY", "K"), mock.patch.object(yc, "OUTPUT_PATH", self.out), \
                mock.patch.object(yc, "load_channels", return_value=["@a", "@b"]), \
                mock.patch.object(yc, "resolve_channel", side_effect=info), \
                mock.patch.object(yc, "list_video_ids", side_effect=listing), \
                mock.patch.object(yc, "get_video_details", side_effect=lambda ids: [self.fake_item(i) for i in ids]):
            yc.main()
        df = pd.read_csv(self.out)
        self.assertEqual(len(df), 4)
        self.assertEqual(df["channel_title"].nunique(), 2)
        for col in ["post_id", "created_time", "post_type", "caption", "views", "likes", "comments", "followers"]:
            self.assertIn(col, df.columns)

    def test_quota_exhausted_midway_keeps_collected_data(self):
        info = lambda ident: {"channel_id": "UC" + ident, "channel_title": ident, "uploads_playlist": "UU",
                              "subscribers": 10}

        def listing(playlist, limit, _calls=[0]):
            _calls[0] += 1
            if _calls[0] == 2:
                raise RuntimeError("Đã hết hạn mức (quota) hôm nay.")
            return ["v1", "v2"]

        with quiet(), mock.patch.object(yc, "API_KEY", "K"), mock.patch.object(yc, "OUTPUT_PATH", self.out), \
                mock.patch.object(yc, "load_channels", return_value=["@a", "@b"]), \
                mock.patch.object(yc, "resolve_channel", side_effect=info), \
                mock.patch.object(yc, "list_video_ids", side_effect=listing), \
                mock.patch.object(yc, "get_video_details", side_effect=lambda ids: [self.fake_item(i) for i in ids]):
            yc.main()
        self.assertEqual(len(pd.read_csv(self.out)), 2)       # dữ liệu kênh đầu vẫn được giữ lại


if __name__ == "__main__":
    unittest.main()