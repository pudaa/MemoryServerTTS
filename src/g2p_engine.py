"""
G2P (Grapheme-to-Phoneme) 引擎抽象层

支持英文 (g2p-en) 和中文 (pypinyin) 的音素转换，
可扩展其他语言的 G2P 后端。
"""

import re
from abc import ABC, abstractmethod


class G2PEngine(ABC):
    """G2P 引擎抽象基类"""

    @abstractmethod
    def word_to_phonemes(self, word: str) -> list[str]:
        """将单个单词转换为音素序列"""
        ...

    @abstractmethod
    def word_to_phoneme_string(self, word: str) -> str:
        """将单个单词转换为音素字符串（用于序列比对）"""
        ...

    def text_to_phonemes(self, text: str) -> list[list[str]]:
        """将文本拆分为单词，返回每个单词的音素序列列表"""
        words = self._tokenize(text)
        return [self.word_to_phonemes(w) for w in words]

    def text_to_word_phoneme_pairs(self, text: str) -> list[tuple[str, list[str]]]:
        """返回 (单词, 音素列表) 的配对列表"""
        words = self._tokenize(text)
        return [(w, self.word_to_phonemes(w)) for w in words]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """通用分词：按空格和标点拆分"""
        return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ\u4e00-\u9fff0-9'\-]+", text.lower())


class EnglishG2P(G2PEngine):
    """英文 G2P 引擎，基于 g2p-en (ARPAbet 音素集)"""

    def __init__(self):
        # g2p-en 依赖 NLTK 的 averaged_perceptron_tagger_eng，确保已下载
        try:
            import nltk
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        except Exception:
            pass

        try:
            from g2p_en import G2p
            self._g2p = G2p()
        except ImportError:
            raise ImportError(
                "g2p-en 未安装，请运行: pip install g2p-en"
            )
        # 重音标记去除映射：HH0→HH, AH1→AH 等
        self._strip_stress = re.compile(r'[0-2]$')

    def word_to_phonemes(self, word: str) -> list[str]:
        """返回去除重音标记的 ARPAbet 音素列表"""
        raw = self._g2p(word)
        # 去重音标记: AH0 → AH, OW1 → OW
        return [self._strip_stress.sub('', p) for p in raw]

    def word_to_phoneme_string(self, word: str) -> str:
        return " ".join(self.word_to_phonemes(word))

    def word_to_phonemes_with_stress(self, word: str) -> list[str]:
        """保留重音标记的音素列表"""
        return self._g2p(word)


class ChineseG2P(G2PEngine):
    """中文 G2P 引擎，基于 pypinyin（拼音作为音素近似）"""

    def __init__(self, with_tone: bool = True):
        try:
            from pypinyin import pinyin, Style
            self._pinyin = pinyin
            self._style = Style.TONE3 if with_tone else Style.NORMAL
        except ImportError:
            raise ImportError(
                "pypinyin 未安装，请运行: pip install pypinyin"
            )
        self._with_tone = with_tone

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """中文按单字拆分"""
        import re
        # 保留中文字符、英文单词、数字
        tokens = re.findall(r'[\u4e00-\u9fff]|[A-Za-z0-9\'\-]+', text.lower())
        return tokens

    def word_to_phonemes(self, word: str) -> list[str]:
        """
        中文"音素"使用拼音声母+韵母拆分：
        'ni3' → ['n', 'i3'] 或保留完整拼音
        """
        result = self._pinyin(word, style=self._style)
        phonemes = []
        for py_list in result:
            for py in py_list:
                if self._with_tone and py and py[-1].isdigit():
                    # 带声调的拼音，拆为声母+韵母
                    initial, final = self._split_initial_final(py)
                    if initial:
                        phonemes.append(initial)
                    if final:
                        phonemes.append(final)
                else:
                    phonemes.append(py)
        return phonemes if phonemes else [word]  # fallback

    def word_to_phoneme_string(self, word: str) -> str:
        return " ".join(self.word_to_phonemes(word))

    @staticmethod
    def _split_initial_final(pinyin: str) -> tuple[str, str]:
        """拆分拼音为声母和韵母，如 'zhuang4' → ('zh', 'uang4')"""
        initials = [
            'zh', 'ch', 'sh',  # 翘舌音先匹配（长前缀）
            'b', 'p', 'm', 'f', 'd', 't', 'n', 'l',
            'g', 'k', 'h', 'j', 'q', 'x',
            'z', 'c', 's', 'r', 'y', 'w',
        ]
        for init in initials:
            if pinyin.startswith(init):
                return init, pinyin[len(init):]
        return '', pinyin


def get_g2p_engine(language: str) -> G2PEngine:
    """工厂函数：根据语言代码获取对应的 G2P 引擎"""
    lang_lower = language.lower() if language else "en"

    if lang_lower in ("en", "english", "eng"):
        return EnglishG2P()
    elif lang_lower in ("zh", "chinese", "chi", "cn", "mandarin"):
        return ChineseG2P()
    else:
        # 默认回退到英文
        print(f"[G2P] 未识别的语言代码 '{language}'，默认使用英文 G2P")
        return EnglishG2P()
