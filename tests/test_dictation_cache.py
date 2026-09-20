"""
词库缓存与预生成器单元测试（Phase 2）

覆盖:
  - cache_key 稳定性 / 区分度（word/voice/language/instruct 变化 → key 不同）
  - gen_config_version 稳定性
  - save/lookup/mark_bad/bump_served/list/delete 往返（临时目录）
  - score_candidate 评分
  - generate_best 择优逻辑（fake synth/verify）

运行（项目根目录）:
    python -m unittest tests.test_dictation_cache -v
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dictation import cache
from src.dictation.generator import score_candidate, generate_best


class FakeConfig:
    """模拟 TTSConfig 中词库缓存涉及的属性"""

    def __init__(self, cache_dir="word-cache"):
        self.model_path = "./models/qwen-1.7b"
        self.short_decode = {
            "temperature": 0.5, "top_k": 20, "top_p": 0.9,
            "repetition_penalty": 1.2, "max_new_tokens": 512, "seed": 42,
        }
        self.verify_max_retries = 3
        self.asr_confidence_threshold = -1.0
        self.word_max_duration_s = 5.0
        self.dictation_conf_threshold = -1.0
        self.dictation_cache_dir = cache_dir
        self.dictation_best_of = 3
        self.dictation_seed_base = 1000
        self.default_language = "English"
        self.default_voice = "aiden"


def _wav(seconds=0.6, sr=16000):
    return np.zeros(int(seconds * sr), dtype=np.float32)


class TestCacheKey(unittest.TestCase):

    def setUp(self):
        self.cfg = FakeConfig()

    def test_stable_same_input(self):
        k1 = cache.cache_key("ahead", "aiden", "English", None, self.cfg)
        k2 = cache.cache_key("ahead", "aiden", "English", None, self.cfg)
        self.assertEqual(k1, k2)

    def test_word_change(self):
        self.assertNotEqual(
            cache.cache_key("ahead", "aiden", "English", None, self.cfg),
            cache.cache_key("behind", "aiden", "English", None, self.cfg))

    def test_voice_change(self):
        self.assertNotEqual(
            cache.cache_key("ahead", "aiden", "English", None, self.cfg),
            cache.cache_key("ahead", "ryan", "English", None, self.cfg))

    def test_instruct_change(self):
        self.assertNotEqual(
            cache.cache_key("ahead", "aiden", "English", None, self.cfg),
            cache.cache_key("ahead", "aiden", "English", "Speak calmly", self.cfg))

    def test_language_change(self):
        self.assertNotEqual(
            cache.cache_key("你好", "vivian", "Chinese", None, self.cfg),
            cache.cache_key("你好", "vivian", "English", None, self.cfg))

    def test_word_case_insensitive(self):
        self.assertEqual(
            cache.cache_key("AHEAD", "aiden", "English", None, self.cfg),
            cache.cache_key("ahead", "aiden", "English", None, self.cfg))

    def test_config_change_invalidates(self):
        k1 = cache.cache_key("ahead", "aiden", "English", None, self.cfg)
        other = FakeConfig()
        other.short_decode = dict(other.short_decode, temperature=0.7)  # 生成配方变更
        k2 = cache.cache_key("ahead", "aiden", "English", None, other)
        self.assertNotEqual(k1, k2)


class TestCacheStore(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cfg = FakeConfig(cache_dir=self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _meta(self, word="ahead"):
        return {
            "word": word, "voice": "aiden", "language": "English",
            "instruct": None, "gen_config_version": "abc123",
            "seed": 1000, "attempts": 1, "verified": True,
            "asr_text": word, "avg_logprob": -0.3,
            "quality_score": 0.9, "duration": 0.6,
        }

    def test_save_lookup_roundtrip(self):
        key = cache.cache_key("ahead", "aiden", "English", None, self.cfg)
        cache.save(self.cfg, key, _wav(), 16000, self._meta())
        entry = cache.lookup(self.cfg, key)
        self.assertIsNotNone(entry)
        self.assertEqual(entry["word"], "ahead")
        self.assertEqual(entry["quality_score"], 0.9)
        self.assertEqual(entry["served_count"], 0)
        self.assertEqual(entry["bad_flags"], [])
        self.assertTrue(os.path.exists(cache._wav_path(self.cfg, key)))

    def test_lookup_missing(self):
        self.assertIsNone(cache.lookup(self.cfg, "nokey"))

    def test_mark_bad_and_recovery(self):
        key = cache.cache_key("ahead", "aiden", "English", None, self.cfg)
        cache.save(self.cfg, key, _wav(), 16000, self._meta())
        entry = cache.mark_bad(self.cfg, key, "emotion")
        self.assertEqual(entry["bad_flags"], ["emotion"])
        self.assertEqual(entry["bad_reason"], "emotion")
        # 重新生成（save 覆盖）自动清除 bad 标记
        cache.save(self.cfg, key, _wav(0.5), 16000, self._meta())
        entry2 = cache.lookup(self.cfg, key)
        self.assertEqual(entry2["bad_flags"], [])
        self.assertIsNone(entry2["bad_reason"])

    def test_bump_served(self):
        key = cache.cache_key("ahead", "aiden", "English", None, self.cfg)
        cache.save(self.cfg, key, _wav(), 16000, self._meta())
        cache.bump_served(self.cfg, key)
        cache.bump_served(self.cfg, key)
        self.assertEqual(cache.lookup(self.cfg, key)["served_count"], 2)

    def test_list_and_delete(self):
        for w in ("ahead", "behind", "cat"):
            key = cache.cache_key(w, "aiden", "English", None, self.cfg)
            cache.save(self.cfg, key, _wav(), 16000, self._meta(w))
        entries = cache.list_entries(self.cfg)
        self.assertEqual(sorted(e["word"] for e in entries),
                         ["ahead", "behind", "cat"])
        s = cache.summary(self.cfg)
        self.assertEqual(s["total"], 3)
        self.assertEqual(s["verified"], 3)
        key = cache.cache_key("ahead", "aiden", "English", None, self.cfg)
        self.assertTrue(cache.delete(self.cfg, key))
        self.assertIsNone(cache.lookup(self.cfg, key))


class TestScoring(unittest.TestCase):

    def test_ideal_window(self):
        self.assertAlmostEqual(score_candidate(-0.0, 0.6), 1.0, places=2)

    def test_low_confidence_low_score(self):
        self.assertLess(score_candidate(-2.0, 0.6), score_candidate(-0.2, 0.6))

    def test_long_duration_penalty(self):
        self.assertLess(score_candidate(-0.2, 4.0), score_candidate(-0.2, 0.6))

    def test_none_conf_score_zero(self):
        self.assertAlmostEqual(score_candidate(None, 0.6), 0.3, places=2)


class TestGenerateBest(unittest.TestCase):

    def test_picks_highest_score(self):
        def synth(text, voice, language, instruct, seed):
            return _wav(), 16000

        confs = iter([-0.8, -0.5, -0.1])   # 第 3 个候选（seed=1002）置信度最高

        def verify(wav, sr, word, language):
            return True, word, next(confs)

        best, failures = generate_best(
            "ahead", "aiden", "English", None, synth, verify,
            best_of=3, seed_base=1000)
        self.assertIsNotNone(best)
        self.assertEqual(best["seed"], 1002)      # 置信度最高者胜出
        self.assertFalse(failures)

    def test_all_failed(self):
        def synth(text, voice, language, instruct, seed):
            return _wav(), 16000

        def verify(wav, sr, word, language):
            return False, "behind", -0.5

        best, failures = generate_best(
            "ahead", "aiden", "English", None, synth, verify,
            best_of=3, seed_base=1000)
        self.assertIsNone(best)
        self.assertEqual(len(failures), 3)
        self.assertTrue(all(f["reason"] == "asr_mismatch" for f in failures))

    def test_duration_out_of_range_skipped(self):
        def synth(text, voice, language, instruct, seed):
            return _wav(seconds=30.0), 16000   # 30s 远超单词合理区间

        def verify(wav, sr, word, language):
            return True, word, -0.2

        best, failures = generate_best(
            "ahead", "aiden", "English", None, synth, verify,
            best_of=2, seed_base=1000)
        self.assertIsNone(best)
        self.assertTrue(all(f["reason"] == "duration_out_of_range"
                            for f in failures))

    def test_low_confidence_skipped(self):
        def synth(text, voice, language, instruct, seed):
            return _wav(), 16000

        def verify(wav, sr, word, language):
            return True, word, -3.0   # 低于 conf_threshold=-1.0

        best, failures = generate_best(
            "ahead", "aiden", "English", None, synth, verify,
            best_of=2, seed_base=1000, conf_threshold=-1.0)
        self.assertIsNone(best)
        self.assertTrue(all(f["reason"] == "low_confidence" for f in failures))

    def test_synth_exception_skipped(self):
        def synth(text, voice, language, instruct, seed):
            raise RuntimeError("boom")

        def verify(wav, sr, word, language):
            return True, word, -0.2

        best, failures = generate_best(
            "ahead", "aiden", "English", None, synth, verify,
            best_of=2, seed_base=1000)
        self.assertIsNone(best)
        self.assertTrue(all(f["reason"].startswith("synth_error")
                            for f in failures))


if __name__ == "__main__":
    unittest.main(verbosity=2)
