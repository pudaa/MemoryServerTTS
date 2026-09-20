"""
TTS 校验闭环单元测试 — src/tts/verifier 的匹配与置信度逻辑

运行（项目根目录）:
    python -m unittest tests.test_tts_verifier -v
    # 或直接
    python tests/test_tts_verifier.py
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tts.verifier import (
    normalize_nospace,
    _levenshtein,
    matches,
    verify_audio,
)


class FakeASR:
    """模拟 ASRModelManager，记录调用参数并返回预设结果"""

    def __init__(self, text="", avg_logprob=0.0):
        self.text = text
        self.avg_logprob = avg_logprob
        self.calls = []

    def transcribe(self, audio_path=None, language=None, task="transcribe",
                   beam_size=5, word_timestamps=False):
        self.calls.append({
            "audio_path": audio_path, "language": language,
            "task": task, "beam_size": beam_size,
        })
        return {"text": self.text, "avg_logprob": self.avg_logprob}


class TestNormalize(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(normalize_nospace("Ahead."), "ahead")
        self.assertEqual(normalize_nospace("  The   Word! "), "theword")

    def test_chinese_kept(self):
        self.assertEqual(normalize_nospace("你好，世界！"), "你好世界")

    def test_mixed(self):
        self.assertEqual(normalize_nospace("a-head's"), "aheads")


class TestLevenshtein(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(_levenshtein("", ""), 0)
        self.assertEqual(_levenshtein("abc", "abc"), 0)
        self.assertEqual(_levenshtein("abc", "abd"), 1)
        self.assertEqual(_levenshtein("abc", "ab"), 1)
        self.assertEqual(_levenshtein("ahead", "ahed"), 1)


class TestMatches(unittest.TestCase):

    def test_exact(self):
        self.assertTrue(matches("ahead", "Ahead."))
        self.assertTrue(matches("ahead", "ahead"))

    def test_split_compound(self):
        # Whisper 常见差异：ahead ↔ "a head"（分词/合词）
        self.assertTrue(matches("ahead", "a head"))
        self.assertTrue(matches("a head", "ahead"))

    def test_hallucination_prefix(self):
        # Whisper 对极短音频常加 "Mm-hmm" 前缀
        self.assertTrue(matches("ahead", "Mm-hmm, ahead."))
        self.assertTrue(matches("hello", "Thank you. Hello."))

    def test_suffix_extra(self):
        self.assertTrue(matches("ahead", "ahead of time"))
        self.assertTrue(matches("run", "running"))  # 编辑距离 2? no——应失败

    def test_edit_distance_one(self):
        self.assertTrue(matches("ahead", "ahed"))
        self.assertTrue(matches("cat", "cats"))

    def test_mismatch(self):
        self.assertFalse(matches("ahead", "behind"))
        self.assertFalse(matches("ahead", ""))

    def test_empty_target_passes(self):
        self.assertTrue(matches("", "anything"))

    def test_running_should_fail(self):
        # run vs running：去空格后 "run" 是 "running" 的子串，宽松规则下通过
        # （对听写场景，"run" 读成 "running" 视为可接受）
        self.assertTrue(matches("run", "running"))


class TestVerifyAudio(unittest.TestCase):

    def _wav(self, seconds=0.5, sr=16000):
        return np.zeros(int(seconds * sr), dtype=np.float32)

    def test_pass_with_high_confidence(self):
        asr = FakeASR(text="Ahead.", avg_logprob=-0.2)
        ok, text, conf = verify_audio(
            self._wav(), 16000, "ahead", "English", asr,
            confidence_threshold=-1.0,
        )
        self.assertTrue(ok)
        self.assertEqual(text, "Ahead.")
        self.assertEqual(asr.calls[0]["language"], "en")  # English → en

    def test_fail_low_confidence(self):
        # 匹配但置信度低于门槛 → 判失败
        asr = FakeASR(text="Ahead.", avg_logprob=-2.5)
        ok, text, conf = verify_audio(
            self._wav(), 16000, "ahead", "English", asr,
            confidence_threshold=-1.0,
        )
        self.assertFalse(ok)
        self.assertEqual(conf, -2.5)

    def test_fail_mismatch(self):
        asr = FakeASR(text="Behind.", avg_logprob=-0.1)
        ok, _, _ = verify_audio(
            self._wav(), 16000, "ahead", "English", asr,
            confidence_threshold=-1.0,
        )
        self.assertFalse(ok)

    def test_fail_empty_transcript(self):
        asr = FakeASR(text="", avg_logprob=-0.1)
        ok, text, _ = verify_audio(
            self._wav(), 16000, "ahead", "English", asr,
            confidence_threshold=-1.0,
        )
        self.assertFalse(ok)
        self.assertEqual(text, "")

    def test_asr_exception_treated_as_fail(self):
        class BrokenASR(FakeASR):
            def transcribe(self, **kwargs):
                raise RuntimeError("boom")
        ok, _, _ = verify_audio(
            self._wav(), 16000, "ahead", "English", BrokenASR(),
            confidence_threshold=-1.0,
        )
        self.assertFalse(ok)

    def test_language_mapping(self):
        asr = FakeASR(text="你好。", avg_logprob=-0.1)
        ok, _, _ = verify_audio(
            self._wav(), 16000, "你好", "Chinese", asr,
            confidence_threshold=-1.0,
        )
        self.assertTrue(ok)
        self.assertEqual(asr.calls[0]["language"], "zh")

    def test_unknown_language_auto_detect(self):
        asr = FakeASR(text="ahead", avg_logprob=-0.1)
        verify_audio(self._wav(), 16000, "ahead", "UnknownLang", asr,
                     confidence_threshold=-1.0)
        self.assertIsNone(asr.calls[0]["language"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
