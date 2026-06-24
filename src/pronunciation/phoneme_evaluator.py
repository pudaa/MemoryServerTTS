"""
基于 G2P + ASR + 音素对齐的发音评价器

工作原理：
1. ASR 将学生录音转写为文字（带单词时间戳）
2. G2P 将参考文本和 ASR 文本分别转为音素序列
3. 使用 SequenceMatcher 对齐参考词与 ASR 词
4. 对每个对齐的词对进行音素级比对
5. 综合计算发音评分

与 PronunciationEvaluator (MFCC+DTW) 的区别：
- 本评价器不需要标准参考音频，只需要参考文本
- 在音素语义层做比对，更贴合语言教学需求
- 可精确定位到具体单词、具体音素的发音错误
"""

import difflib
import tempfile
import os
from pathlib import Path
from typing import Optional

from src.pronunciation.g2p_engine import G2PEngine, get_g2p_engine


class PhonemeEvaluator:
    """音素级发音评价器"""

    def __init__(self, asr_model, language: str = "en"):
        """
        Args:
            asr_model: ASRModelManager 实例
            language: 默认语言代码
        """
        self.asr_model = asr_model
        self.default_language = language
        # 缓存 G2P 引擎，按语言懒加载
        self._g2p_cache: dict[str, G2PEngine] = {}

    def _get_g2p(self, language: str | None) -> G2PEngine:
        lang = language or self.default_language
        if lang not in self._g2p_cache:
            self._g2p_cache[lang] = get_g2p_engine(lang)
        return self._g2p_cache[lang]

    def evaluate(
        self,
        audio_path: str,
        reference_text: str,
        language: str | None = None,
    ) -> dict:
        """
        核心评价方法：对比学生录音与参考文本的发音

        Args:
            audio_path: 学生录音文件路径
            reference_text: 参考文本（期望朗读内容）
            language: 语言代码（如 'en', 'zh'）

        Returns:
            {
                "overall_score": float,       # 综合评分 0-100
                "phoneme_accuracy": float,     # 音素准确率
                "word_count_reference": int,   # 参考文本单词数
                "word_count_spoken": int,      # 实际说出的单词数
                "words": [                     # 逐词评分
                    {
                        "word": str,                   # 参考词
                        "spoken_word": str | None,     # 学生实际说的词
                        "score": float,                # 该词评分
                        "expected_phonemes": [str],    # 期望音素
                        "actual_phonemes": [str],      # 实际音素
                        "phoneme_accuracy": float,     # 该词音素准确率
                        "errors": [                    # 音素错误详情
                            {
                                "type": "substitution|deletion|insertion",
                                "expected": str | None,
                                "actual": str | None,
                                "position": int,
                            }
                        ],
                        "status": "correct|mispronounced|missing|extra",
                    }
                ],
                "level": str,                  # excellent/good/fair/poor/very_poor
                "feedback": str,               # 中文反馈建议
            }
        """
        g2p = self._get_g2p(language)

        # ── 第一步：ASR 转录学生录音 ──
        asr_result = self.asr_model.transcribe(
            audio_path=audio_path,
            word_timestamps=True,
            language=language,  # 透传语言代码给 Faster-Whisper
        )

        spoken_text = asr_result.get("text", "")
        spoken_words_raw = self._extract_words_from_asr(asr_result)

        # ── 第二步：G2P 处理参考文本 ──
        ref_word_phonemes = g2p.text_to_word_phoneme_pairs(reference_text)
        # ref_words: 参考文本的单词列表
        # ref_phonemes: 对应的音素列表

        # ── 第三步：G2P 处理 ASR 输出 ──
        asr_word_phonemes = g2p.text_to_word_phoneme_pairs(spoken_text)

        # ── 第四步：词级对齐 ──
        ref_words_only = [item[0] for item in ref_word_phonemes]
        asr_words_only = [item[0] for item in asr_word_phonemes]

        alignment = self._align_word_sequences(ref_words_only, asr_words_only)

        # ── 第五步：逐词音素比对 ──
        word_results = []
        total_phoneme_correct = 0
        total_phoneme_expected = 0

        for ref_idx, asr_idx in alignment:
            if ref_idx is not None and asr_idx is not None:
                # 匹配对：比较音素
                ref_word, ref_phons = ref_word_phonemes[ref_idx]
                asr_word, asr_phons = asr_word_phonemes[asr_idx]

                ph_acc, errors = self._compare_phonemes(ref_phons, asr_phons, ref_word)

                correct_in_word = len(ref_phons) - sum(1 for e in errors if e["type"] != "insertion")
                word_score = (correct_in_word / max(len(ref_phons), 1)) * 100

                total_phoneme_correct += correct_in_word
                total_phoneme_expected += len(ref_phons)

                status = "correct" if word_score >= 80 else "mispronounced"
                if ref_word.lower() != asr_word.lower():
                    status = "mispronounced"  # 词本身不一致

                word_results.append({
                    "word": ref_word,
                    "spoken_word": asr_word,
                    "start_time": self._find_word_time(spoken_words_raw, asr_idx),
                    "end_time": self._find_word_time(spoken_words_raw, asr_idx, is_end=True),
                    "score": round(word_score, 1),
                    "expected_phonemes": ref_phons,
                    "actual_phonemes": asr_phons,
                    "phoneme_accuracy": round(correct_in_word / max(len(ref_phons), 1), 3),
                    "errors": errors,
                    "status": status,
                })

            elif ref_idx is not None:
                # 学生没读这个词（deletion）
                ref_word, ref_phons = ref_word_phonemes[ref_idx]
                total_phoneme_expected += len(ref_phons)
                word_results.append({
                    "word": ref_word,
                    "spoken_word": None,
                    "start_time": None,
                    "end_time": None,
                    "score": 0.0,
                    "expected_phonemes": ref_phons,
                    "actual_phonemes": [],
                    "phoneme_accuracy": 0.0,
                    "errors": [{"type": "deletion", "expected": p, "actual": None, "position": i}
                               for i, p in enumerate(ref_phons)],
                    "status": "missing",
                })

            elif asr_idx is not None:
                # 学生多读了词（insertion），不计入评分但记录
                asr_word, asr_phons = asr_word_phonemes[asr_idx]
                word_results.append({
                    "word": None,
                    "spoken_word": asr_word,
                    "start_time": self._find_word_time(spoken_words_raw, asr_idx),
                    "end_time": self._find_word_time(spoken_words_raw, asr_idx, is_end=True),
                    "score": 0.0,
                    "expected_phonemes": [],
                    "actual_phonemes": asr_phons,
                    "phoneme_accuracy": 0.0,
                    "errors": [{"type": "insertion", "expected": None, "actual": p, "position": i}
                               for i, p in enumerate(asr_phons)],
                    "status": "extra",
                })

        # ── 第六步：综合评分 ──
        if total_phoneme_expected > 0:
            phoneme_accuracy = total_phoneme_correct / total_phoneme_expected
        else:
            phoneme_accuracy = 0.0

        # 综合评分：音素准确率 * 100，考虑缺失词惩罚
        ref_word_count = len([w for w in word_results if w["status"] != "extra"]) # 参考文本的单词数（不算多余词）
        missing_count = len([w for w in word_results if w["status"] == "missing"])
        extra_count = len([w for w in word_results if w["status"] == "extra"])

        if ref_word_count > 0:
            # 基础分 = 音素准确率 * 100
            base_score = phoneme_accuracy * 100
            # 缺失惩罚：每个缺失词扣分
            missing_penalty = (missing_count / ref_word_count) * 20
            # 多余词惩罚
            extra_penalty = min(extra_count * 2, 10)
            overall_score = max(0, base_score - missing_penalty - extra_penalty)
        else:
            overall_score = 0.0

        overall_score = round(overall_score, 1)

        return {
            "overall_score": overall_score,
            "phoneme_accuracy": round(phoneme_accuracy, 3),
            "word_count_reference": sum(1 for w in word_results if w["status"] != "extra"),
            "word_count_spoken": len(spoken_words_raw),
            "words": word_results,
            "asr_transcript": spoken_text,
            "reference_text": reference_text,
            "level": self._get_level(overall_score),
            "feedback": self._generate_feedback(overall_score, word_results),
        }

    # ─── 内部方法 ───

    def _extract_words_from_asr(self, asr_result: dict) -> list[dict]:
        """从 ASR 结果中提取单词列表（带时间戳）"""
        words = []
        for seg in asr_result.get("segments", []):
            for w in seg.get("words", []):
                words.append({
                    "word": w["word"],
                    "start": w["start"],
                    "end": w["end"],
                    "probability": w["probability"],
                })
        return words

    def _find_word_time(self, spoken_words: list[dict], idx: int,
                        is_end: bool = False) -> float | None:
        """获取某个单词在音频中的时间"""
        if spoken_words and 0 <= idx < len(spoken_words):
            return spoken_words[idx]["end"] if is_end else spoken_words[idx]["start"]
        return None

    @staticmethod
    def _align_word_sequences(
        ref_words: list[str], asr_words: list[str]
    ) -> list[tuple[int | None, int | None]]:
        """
        使用 difflib.SequenceMatcher 对齐两个单词序列

        Returns:
            [(ref_idx, asr_idx), ...]
            None 表示该侧无对应（插入或删除）
        """
        matcher = difflib.SequenceMatcher(
            None,
            [w.lower() for w in ref_words],
            [w.lower() for w in asr_words],
        )

        alignment = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                # 一对一匹配
                for k in range(i2 - i1):
                    alignment.append((i1 + k, j1 + k))
            elif tag == "replace":
                # 替换：尝试最佳配对
                max_len = max(i2 - i1, j2 - j1)
                for k in range(max_len):
                    ref_i = i1 + k if k < (i2 - i1) else None
                    asr_j = j1 + k if k < (j2 - j1) else None
                    alignment.append((ref_i, asr_j))
            elif tag == "delete":
                # 参考中有、ASR 中没有
                for k in range(i1, i2):
                    alignment.append((k, None))
            elif tag == "insert":
                # ASR 中有、参考中没有
                for k in range(j1, j2):
                    alignment.append((None, k))

        return alignment

    @staticmethod
    def _compare_phonemes(
        expected: list[str], actual: list[str], word: str = ""
    ) -> tuple[float, list[dict]]:
        """
        比较两个音素序列，返回 (音素准确率, 错误列表)

        使用编辑距离对齐，检测 substitution/deletion/insertion
        """
        matcher = difflib.SequenceMatcher(None, expected, actual)
        errors = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue  # 音素匹配正确
            elif tag == "replace":
                for k in range(max(i2 - i1, j2 - j1)):
                    exp = expected[i1 + k] if i1 + k < i2 else None
                    act = actual[j1 + k] if j1 + k < j2 else None
                    if exp is None:
                        errors.append({"type": "insertion", "expected": None,
                                       "actual": act, "position": i1 + k})
                    elif act is None:
                        errors.append({"type": "deletion", "expected": exp,
                                       "actual": None, "position": i1 + k})
                    else:
                        errors.append({"type": "substitution", "expected": exp,
                                       "actual": act, "position": i1 + k})
            elif tag == "delete":
                for k in range(i1, i2):
                    errors.append({"type": "deletion", "expected": expected[k],
                                   "actual": None, "position": k})
            elif tag == "insert":
                for k in range(j1, j2):
                    errors.append({"type": "insertion", "expected": None,
                                   "actual": actual[k], "position": j1 + k})

        accuracy = (len(expected) - sum(1 for e in errors if e["type"] != "insertion")) / max(len(expected), 1)
        return accuracy, errors

    @staticmethod
    def _get_level(score: float) -> str:
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "fair"
        elif score >= 40:
            return "poor"
        else:
            return "very_poor"

    @staticmethod
    def _generate_feedback(score: float, word_results: list[dict]) -> str:
        """生成中文反馈建议"""
        total = len(word_results)
        problem_words = [w for w in word_results
                         if w["status"] in ("mispronounced", "missing")]

        if score >= 90:
            feedback = "发音非常标准！"
        elif score >= 75:
            feedback = "发音良好。"
        elif score >= 60:
            feedback = "发音基本正确，但还有提升空间。"
        elif score >= 40:
            feedback = "发音有待改进。"
        else:
            feedback = "发音需要大幅提升。建议从基础音标开始练习。"

        if problem_words:
            word_list = [w["word"] or w.get("spoken_word", "?") for w in problem_words[:5]]
            feedback += f" 需重点练习的词汇：{', '.join(word_list)}"
            if len(problem_words) > 5:
                feedback += f" 等共 {len(problem_words)} 个词。"

        return feedback

    def batch_evaluate(
        self,
        pairs: list[dict],
        language: str | None = None,
    ) -> list[dict]:
        """
        批量评价

        Args:
            pairs: [{"audio": "path/to/audio.wav", "reference_text": "hello world"}, ...]

        Returns:
            评价结果列表
        """
        results = []
        for pair in pairs:
            try:
                result = self.evaluate(
                    audio_path=pair["audio"],
                    reference_text=pair["reference_text"],
                    language=language,
                )
                result["audio"] = pair["audio"]
                results.append(result)
            except Exception as e:
                results.append({
                    "audio": pair.get("audio", ""),
                    "error": str(e),
                })
        return results
