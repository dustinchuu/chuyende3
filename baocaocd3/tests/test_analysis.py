# -*- coding: utf-8 -*-
"""Kiểm thử file phân tích: hàm nhỏ, làm sạch, kiểm định thống kê, kiểm soát kênh, mô hình và chạy trọn vẹn.

Chạy nhanh (bỏ bài chạy trọn vẹn, mất ~30-60 giây):  set SKIP_SLOW=1   (Windows)  /  export SKIP_SLOW=1
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from helpers import load_analysis, make_row, quiet, raw_frame, synthetic_raw

sma = load_analysis()
SKIP_SLOW = os.environ.get("SKIP_SLOW") == "1"


def cleaned(raw):
    with quiet():
        df = sma.clean_data(raw)
        return sma.add_channel_adjusted(df)


class OutputDirMixin:
    """Chuyển thư mục kết quả sang thư mục tạm để không ghi đè file thật."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = self.tmp.name
        self.fig = os.path.join(self.out, "figures")
        os.makedirs(self.fig, exist_ok=True)
        self.patches = [mock.patch.object(sma, "OUTPUT_DIR", self.out), mock.patch.object(sma, "FIG_DIR", self.fig)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()


# ----------------------------------------------------------------------
class TestParsing(unittest.TestCase):
    def test_parse_count_suffixes(self):
        self.assertEqual(sma.parse_count("1.2K"), 1200)
        self.assertEqual(sma.parse_count("3M"), 3_000_000)
        self.assertEqual(sma.parse_count("1,234"), 1234)
        self.assertEqual(sma.parse_count("  50 "), 50)

    def test_parse_count_passthrough_and_invalid(self):
        self.assertEqual(sma.parse_count(42), 42)
        self.assertEqual(sma.parse_count(3.5), 3.5)
        self.assertTrue(np.isnan(sma.parse_count(np.nan)))
        self.assertTrue(np.isnan(sma.parse_count(None)))
        self.assertTrue(np.isnan(sma.parse_count("N/A")))
        self.assertTrue(np.isnan(sma.parse_count("abc")))

    def test_parse_datetime_mixed_formats(self):
        s = pd.Series(["2025-03-04 10:30:00", "05/03/2025 08:00", "không rõ"])
        out = sma.parse_datetime(s)
        self.assertEqual(out.iloc[0], pd.Timestamp("2025-03-04 10:30:00"))
        self.assertEqual(out.iloc[1], pd.Timestamp("2025-03-05 08:00:00"))      # dd/mm/yyyy
        self.assertTrue(pd.isna(out.iloc[2]))

    def test_clean_text(self):
        t = sma.clean_text("Xem NGAY!!! https://abc.com/x @user #Tips 2025 hay quá?")
        self.assertEqual(t, "xem ngay hay quá")
        self.assertNotIn("http", t)
        self.assertNotIn("tips", t)


# ----------------------------------------------------------------------
class TestStatHelpers(unittest.TestCase):
    def test_holm_known_values(self):
        adj = sma.holm([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])

    def test_holm_properties(self):
        p = np.array([0.001, 0.2, 0.5, 0.04])
        adj = sma.holm(p)
        self.assertTrue(np.all(adj >= p))              # không bao giờ nhỏ hơn p gốc
        self.assertTrue(np.all(adj <= 1))
        self.assertEqual(list(np.argsort(adj, kind="stable")), list(np.argsort(p, kind="stable")))  # giữ thứ tự

    def test_holm_single_value_unchanged(self):
        np.testing.assert_allclose(sma.holm([0.03]), [0.03])

    def test_magnitude_thresholds(self):
        self.assertEqual(sma.magnitude(0.05, "r"), "không đáng kể")
        self.assertEqual(sma.magnitude(-0.2, "r"), "nhỏ")
        self.assertEqual(sma.magnitude(0.4, "r"), "trung bình")
        self.assertEqual(sma.magnitude(0.8, "r"), "lớn")
        self.assertEqual(sma.magnitude(0.005, "eps"), "không đáng kể")
        self.assertEqual(sma.magnitude(0.2, "eps"), "lớn")

    def test_eta_squared_extremes(self):
        self.assertAlmostEqual(sma.eta_squared([1, 1, 5, 5], ["a", "a", "b", "b"]), 1.0)
        self.assertAlmostEqual(sma.eta_squared([1, 5, 1, 5], ["a", "a", "b", "b"]), 0.0)
        self.assertEqual(sma.eta_squared([3, 3, 3], ["a", "b", "c"]), 0.0)        # không có biến thiên


class TestRunTest(unittest.TestCase):
    def frame(self, shift=0.0, n=200, seed=1):
        rng = np.random.default_rng(seed)
        flag = np.array([True] * n + [False] * n)
        y = rng.normal(0, 1, 2 * n) + np.where(flag, shift, 0)
        return pd.DataFrame({"y": y, "flag": flag, "g": np.where(flag, "x", "y")}), flag

    def test_binary_detects_real_difference(self):
        df, flag = self.frame(shift=1.5)
        r = sma.run_test(dict(kind="binary", a=pd.Series(flag), b=pd.Series(~flag)), df, "y")
        self.assertLess(r["p"], 1e-6)
        self.assertGreater(r["eff"], 0.5)           # nhóm A cao hơn
        self.assertEqual(r["sign"], 1)

    def test_binary_no_difference(self):
        df, flag = self.frame(shift=0.0)
        r = sma.run_test(dict(kind="binary", a=pd.Series(flag), b=pd.Series(~flag)), df, "y")
        self.assertGreater(r["p"], 0.05)
        self.assertLess(abs(r["eff"]), 0.2)

    def test_binary_direction_flips_with_groups(self):
        df, flag = self.frame(shift=1.5)
        r = sma.run_test(dict(kind="binary", a=pd.Series(~flag), b=pd.Series(flag)), df, "y")
        self.assertLess(r["eff"], -0.5)
        self.assertEqual(r["sign"], -1)

    def test_too_small_group_returns_none(self):
        df, flag = self.frame(n=10)
        self.assertIsNone(sma.run_test(dict(kind="binary", a=pd.Series(flag), b=pd.Series(~flag)), df, "y"))

    def test_multi_group(self):
        rng = np.random.default_rng(2)
        df = pd.DataFrame({"g": np.repeat(["a", "b", "c"], 60),
                           "y": np.concatenate([rng.normal(m, 1, 60) for m in (0, 2, 4)])})
        r = sma.run_test(dict(kind="multi", col="g"), df, "y")
        self.assertLess(r["p"], 1e-6)
        self.assertGreater(r["eff"], 0.14)

    def test_constant_column_is_skipped_not_crash(self):
        """Hồi quy: cột không đổi từng làm Spearman trả NaN rồi chương trình sập."""
        df = pd.DataFrame({"x": [5] * 100, "y": np.random.default_rng(0).normal(size=100)})
        self.assertIsNone(sma.run_test(dict(kind="corr", x="x"), df, "y"))

    def test_correlation(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=200)
        df = pd.DataFrame({"x": x, "y": 2 * x + rng.normal(0, 0.3, 200)})
        r = sma.run_test(dict(kind="corr", x="x"), df, "y")
        self.assertGreater(r["eff"], 0.9)
        self.assertLess(r["p"], 1e-6)


# ----------------------------------------------------------------------
class TestCleanData(unittest.TestCase):
    def base_rows(self, n=60):
        return [make_row(i, days=i) for i in range(n)]

    def ids(self, df):
        return set(df["post_id"])

    def test_computes_rate_and_features(self):
        rows = self.base_rows() + [make_row(900, rate=0.03, views=20000, hour=20, caption="Hay không? #a #b")]
        df = cleaned(raw_frame(rows))
        r = df[df.post_id == "A00900"].iloc[0]
        self.assertAlmostEqual(r["engagement_rate"], (600 + 60) / 20000 * 100)     # (like + comment) / view * 100
        self.assertEqual(r["n_hashtags"], 2)
        self.assertEqual(r["has_question"], 1)
        self.assertEqual(r["hour"], 20)
        self.assertEqual(r["time_band"], "Tối (18-23h)")
        self.assertEqual(r["shares"], 0)

    def test_removes_invalid_rows(self):
        recent = (pd.Timestamp.now() - pd.Timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")
        bad = [
            make_row(901, created="không rõ"),                  # thời gian không hợp lệ
            make_row(902, created=recent),                      # quá mới
            make_row(903),                                      # ẩn lượt thích (đặt NaN bên dưới)
            make_row(904, views=0),                             # 0 lượt xem
            make_row(905, views=100, likes=500, comments=0),    # like > view
            make_row(906, duration=0),                          # thời lượng 0
        ]
        rows = self.base_rows() + bad
        rows.append(dict(rows[0]))                              # trùng post_id
        raw = raw_frame(rows)
        raw.loc[raw.post_id == "A00903", "likes"] = np.nan
        out = cleaned(raw)
        self.assertEqual(len(out), 60)
        for i in range(901, 907):
            self.assertNotIn(f"A{i:05d}", self.ids(out))
        self.assertEqual(out["post_id"].is_unique, True)

    def test_disabled_comments_become_zero_not_dropped(self):
        df = raw_frame(self.base_rows() + [make_row(907)])
        df.loc[df.post_id == "A00907", "comments"] = np.nan
        out = cleaned(df)
        row = out[out.post_id == "A00907"].iloc[0]
        self.assertEqual(row["comments"], 0)

    def test_winsorize_caps_rate_not_counts(self):
        """Hồi quy: trước đây cắt ngưỡng trên SỐ ĐẾM nên kênh lớn bị thiệt. Số đếm phải được giữ nguyên."""
        rng = np.random.default_rng(5)
        small = [make_row(i, channel="S", rate=0.05, views=int(rng.integers(15000, 25000)), days=i % 200)
                 for i in range(300)]
        big = [make_row(i, channel="B", rate=0.05, views=int(rng.integers(18_000_000, 22_000_000)), days=i % 200)
               for i in range(5)]
        raw = raw_frame(small + big)
        out = cleaned(raw)
        merged = out.merge(raw[["post_id", "likes", "comments"]], on="post_id", suffixes=("", "_raw"))
        self.assertTrue((merged["likes"] == merged["likes_raw"]).all(), "số like bị cắt ngưỡng")
        self.assertTrue((merged["comments"] == merged["comments_raw"]).all(), "số comment bị cắt ngưỡng")
        big_rate = out[out.channel_title == "Kenh B"]["target_metric"].median()
        small_rate = out[out.channel_title == "Kenh S"]["target_metric"].median()
        self.assertAlmostEqual(big_rate, small_rate, delta=0.05)     # cùng tỷ lệ thì không bị lệch theo quy mô

    def test_extreme_rate_is_capped(self):
        rows = self.base_rows(300) + [make_row(950, views=1000, likes=900, comments=90)]   # tỷ lệ 99%
        out = cleaned(raw_frame(rows))
        self.assertLess(out["target_metric"].max(), 50)

    def test_output_sorted_by_time(self):
        out = cleaned(raw_frame(self.base_rows()[::-1]))
        self.assertTrue(out["created_time"].is_monotonic_increasing)


class TestLoadData(unittest.TestCase):
    def test_missing_file_exits(self):
        with quiet(), self.assertRaises(SystemExit):
            sma.load_data("/khong/ton/tai.csv")

    def test_missing_columns_exits(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.csv")
            pd.DataFrame({"a": [1]}).to_csv(path, index=False)
            with quiet(), self.assertRaises(SystemExit):
                sma.load_data(path)

    def test_valid_file_loads(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.csv")
            raw_frame([make_row(i) for i in range(5)]).to_csv(path, index=False)
            with quiet():
                df = sma.load_data(path)
            self.assertEqual(len(df), 5)


# ----------------------------------------------------------------------
class TestChannelAdjustment(unittest.TestCase):
    def setUp(self):
        self.df = cleaned(synthetic_raw([("A", 0.02), ("B", 0.06), ("C", 0.04)], 80, seed=4))

    def test_has_channels(self):
        self.assertTrue(sma.has_channels(self.df))
        self.assertFalse(sma.has_channels(self.df.assign(channel_title="x")))
        self.assertFalse(sma.has_channels(self.df.drop(columns=["channel_title"])))

    def test_metric_adj_centered_per_channel(self):
        means = self.df.groupby("channel_title")["metric_adj"].mean()
        np.testing.assert_allclose(means.values, 0, atol=1e-9)

    def test_rel_metric_median_is_one_per_channel(self):
        med = self.df.groupby("channel_title")["rel_metric"].median()
        np.testing.assert_allclose(med.values, 1.0, atol=1e-9)

    def test_analysis_metric_switches_with_number_of_channels(self):
        self.assertEqual(sma.ANALYSIS_COL, "rel_metric")
        single = self.df[self.df.channel_title == "Kenh A"].copy()
        sma.add_channel_adjusted(single)
        self.assertEqual(sma.ANALYSIS_COL, "target_metric")
        sma.add_channel_adjusted(self.df)                       # trả lại trạng thái cho các bài sau
        self.assertEqual(sma.ANALYSIS_COL, "rel_metric")

    def test_channel_consistency_counts_agreeing_channels(self):
        t = dict(kind="binary", a=self.df["is_weekend"] == 1, b=self.df["is_weekend"] == 0)
        txt = sma.channel_consistency(t, self.df, 1)
        self.assertRegex(txt, r"^\d/3 kênh$")
        self.assertEqual(sma.channel_consistency(t, self.df, 0), "")       # không có chiều chung -> bỏ qua
        self.assertEqual(sma.channel_consistency(dict(kind="multi", col="post_type"), self.df, 1), "")


# ----------------------------------------------------------------------
class TestStatisticalTests(OutputDirMixin, unittest.TestCase):
    def row(self, table, prefix):
        return table[table["Giả thuyết"].str.startswith(prefix)].iloc[0]

    def test_confounder_effect_disappears_after_channel_control(self):
        """Kênh thấp đăng nhiều video ngắn, kênh cao đăng nhiều video dài; KHÔNG có hiệu ứng thật của loại video."""
        raw = synthetic_raw([("Low", 0.02), ("High", 0.06)], 250, seed=7, short_share=[0.85, 0.15])
        df = cleaned(raw)
        with quiet():
            table = sma.statistical_tests(df)
        h2 = self.row(table, "H2")
        self.assertLess(h2["p Holm (gốc)"], 0.01)                          # nhìn gộp tưởng là có khác biệt
        self.assertGreater(h2["p Holm (kiểm soát kênh)"], 0.05)            # nhưng thực ra là do kênh
        self.assertIn("MẤT", h2["Kết luận"])

    def test_real_effect_survives_channel_control(self):
        raw = synthetic_raw([("A", 0.02), ("B", 0.06), ("C", 0.04)], 200, seed=8, evening_boost=1.3)
        df = cleaned(raw)
        with quiet():
            table = sma.statistical_tests(df)
        h3 = self.row(table, "H3")
        self.assertLess(h3["p Holm (gốc)"], 0.05)
        self.assertLess(h3["p Holm (kiểm soát kênh)"], 0.05)
        self.assertGreater(h3["Hiệu ứng (kiểm soát kênh)"], 0)             # tối cao hơn
        self.assertIn("vẫn giữ", h3["Kết luận"])
        self.assertEqual(h3["Cùng chiều theo kênh"], "3/3 kênh")

    def test_writes_csv_and_has_expected_columns(self):
        df = cleaned(synthetic_raw([("A", 0.03), ("B", 0.05)], 120, seed=9))
        with quiet():
            table = sma.statistical_tests(df)
        self.assertTrue(os.path.exists(os.path.join(self.out, "statistical_tests.csv")))
        for col in ["Giả thuyết", "p Holm (gốc)", "p Holm (kiểm soát kênh)", "Kết luận"]:
            self.assertIn(col, table.columns)
        self.assertTrue(((table["p Holm (gốc)"] >= 0) & (table["p Holm (gốc)"] <= 1)).all())

    def test_single_channel_skips_adjustment(self):
        df = cleaned(synthetic_raw([("A", 0.03)], 200, seed=10))
        with quiet():
            table = sma.statistical_tests(df)
        self.assertTrue(table["p Holm (kiểm soát kênh)"].isna().all())

    def test_channel_analysis_outputs(self):
        df = cleaned(synthetic_raw([("Low", 0.02), ("High", 0.06)], 150, seed=7, short_share=[0.85, 0.15]))
        with quiet():
            sma.channel_analysis(df)
        self.assertTrue(os.path.exists(os.path.join(self.out, "channel_summary.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.fig, "15_kenh_x_hinh_thuc.png")))


class TestContentAndTime(OutputDirMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.df = cleaned(synthetic_raw([("A", 0.02), ("B", 0.06)], 200, seed=11))

    def test_content_tables_use_channel_relative_metric(self):
        with quiet():
            res = sma.analyze_content(self.df)
        by_type = res["by_type"]
        self.assertEqual(set(by_type.index), {"short", "long"})
        self.assertIn("chi_so_trung_vi", by_type.columns)
        self.assertTrue((by_type["chi_so_trung_vi"].between(0.8, 1.2)).all())     # quanh 1 vì đã chia cho mức kênh
        self.assertEqual(len(res["top_posts"]), 10)
        self.assertIn("rel_metric", res["top_posts"].columns)

    def test_time_heatmap_uses_bands_not_hours(self):
        with quiet():
            res = sma.analyze_time(self.df)
        self.assertEqual(list(res["heatmap"].columns), sma.TIME_BANDS)
        self.assertEqual(res["heatmap"].shape[1], 5)
        self.assertEqual(len(res["by_hour"]), 24)


# ----------------------------------------------------------------------
class TestModel(unittest.TestCase):
    def frame(self, n=300, signal=True, seed=0):
        rng = np.random.default_rng(seed)
        hour = rng.integers(0, 24, n)
        y = (2.0 + (1.5 * (hour >= 18) if signal else 0) + rng.normal(0, 0.2, n))
        return pd.DataFrame({"hour": hour, "post_type": rng.choice(["short", "long"], n)}), pd.Series(y)

    def test_model_beats_baseline_when_signal_exists(self):
        X, y = self.frame(signal=True)
        base, _, _ = sma.evaluate_model("b", sma.make_regressors()[sma.BASELINE], X, y, 240, ["hour"])
        lin, _, _ = sma.evaluate_model("l", sma.make_regressors()["Linear Regression"], X, y, 240, ["hour"])
        self.assertLess(base["R2 (log)"], 0.05)
        self.assertGreater(lin["R2 (log)"], base["R2 (log)"] + 0.3)

    def test_baseline_is_not_better_than_noise_level(self):
        X, y = self.frame(signal=False, seed=3)
        row, _, _ = sma.evaluate_model("b", sma.make_regressors()[sma.BASELINE], X, y, 240, ["hour"])
        self.assertLess(row["R2 (log)"], 0.05)

    def test_evaluate_model_returns_expected_keys(self):
        X, y = self.frame()
        row, pipe, pred = sma.evaluate_model("l", sma.make_regressors()["Linear Regression"], X, y, 240, ["hour"])
        for k in ["R2 (log)", "R2 CV theo thời gian (TB)", "MAE (đơn vị thước đo)", "RMSE (đơn vị thước đo)"]:
            self.assertIn(k, row)
        self.assertEqual(len(pred), 60)
        self.assertTrue((pred > 0).all())                    # dự đoán đã đổi ngược từ log về đơn vị gốc

    def test_make_regressors_contains_baseline(self):
        self.assertIn(sma.BASELINE, sma.make_regressors())


@unittest.skipIf(SKIP_SLOW, "bỏ qua bài chạy trọn vẹn (SKIP_SLOW=1)")
class TestEndToEnd(OutputDirMixin, unittest.TestCase):
    def test_main_runs_and_produces_all_outputs(self):
        raw = synthetic_raw([("A", 0.02), ("B", 0.06), ("C", 0.04), ("D", 0.03), ("E", 0.05)], 130,
                            seed=12, evening_boost=1.25)
        csv = os.path.join(self.out, "in.csv")
        raw.to_csv(csv, index=False)
        with quiet(), mock.patch.object(sys, "argv", ["prog", csv]):
            sma.main()
        for name in ["posts_cleaned.csv", "statistical_tests.csv", "channel_summary.csv", "model_metrics.csv",
                     "model_unseen_channels.csv", "top_posts.csv"]:
            self.assertTrue(os.path.exists(os.path.join(self.out, name)), f"thiếu {name}")
        self.assertEqual(len([f for f in os.listdir(self.fig) if f.endswith(".png")]), 15)
        metrics = pd.read_csv(os.path.join(self.out, "model_metrics.csv"), index_col=0)
        self.assertIn(sma.BASELINE, metrics.index)
        self.assertIn("Random Forest", metrics.index)
        cleaned_csv = pd.read_csv(os.path.join(self.out, "posts_cleaned.csv"))
        self.assertEqual(len(cleaned_csv), 650)
        self.assertFalse(cleaned_csv["target_metric"].isna().any())

    def test_main_exits_cleanly_without_data_file(self):
        with quiet(), mock.patch.object(sys, "argv", ["prog", os.path.join(self.out, "none.csv")]):
            with self.assertRaises(SystemExit):
                sma.main()


if __name__ == "__main__":
    unittest.main()