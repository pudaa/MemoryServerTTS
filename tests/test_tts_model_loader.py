"""
短文本判定单元测试 — src/tts/model_loader.is_single_word

覆盖：单个单词（含补标点后）、中文词、短句与多词短语排除、
超长兜底、空串等边界。

运行（项目根目录）:
    python -m unittest tests.test_tts_model_loader -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tts.model_loader import is_single_word


class TestIsSingleWord(unittest.TestCase):

    def test_single_english_word(self):
        self.assertTrue(is_single_word("ahead"))
        self.assertTrue(is_single_word("well"))
        self.assertTrue(is_single_word("hello"))

    def test_with_appended_period(self):
        # 服务端短文本修复会补句号："Ahead."
        self.assertTrue(is_single_word("Ahead."))
        self.assertTrue(is_single_word("ahead."))

    def test_apostrophe_and_hyphen(self):
        self.assertTrue(is_single_word("don't"))
        self.assertTrue(is_single_word("a-head"))

    def test_number(self):
        self.assertTrue(is_single_word("2024"))

    def test_chinese_word(self):
        self.assertTrue(is_single_word("你好"))
        self.assertTrue(is_single_word("你好。"))

    def test_short_sentence_rejected(self):
        # 用户报告的案例：短句不应判定为单词
        self.assertFalse(is_single_word("Hello. This is a speed benchmark test."))
        self.assertFalse(is_single_word("How are you?"))
        self.assertFalse(is_single_word("in the morning"))

    def test_multiword_phrase_rejected(self):
        self.assertFalse(is_single_word("a head"))
        self.assertFalse(is_single_word("apple pie"))

    def test_empty_and_whitespace(self):
        self.assertFalse(is_single_word(""))
        self.assertFalse(is_single_word("   "))
        self.assertFalse(is_single_word("..."))

    def test_length_cap(self):
        self.assertFalse(is_single_word("a" * 41))
        self.assertTrue(is_single_word("a" * 40))

    def test_internal_punctuation_rejected(self):
        # 句内标点（非尾部）说明是多词句子
        self.assertFalse(is_single_word("hello, world"))
        self.assertFalse(is_single_word("well, well"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
