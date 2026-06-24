import librosa
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from typing import Optional, Dict, List


class PronunciationEvaluator:
    """发音评价器，基于MFCC + DTW算法"""

    def __init__(self, sr: int = 16000, n_mfcc: int = 13):
        """
        初始化发音评价器
        
        Args:
            sr: 采样率
            n_mfcc: MFCC特征维度
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.max_distance = 1000  # 归一化参数，可根据实际调整

    def extract_mfcc(self, audio_path: str, n_mfcc: Optional[int] = None) -> np.ndarray:
        """
        提取MFCC特征
        
        Args:
            audio_path: 音频文件路径
            n_mfcc: MFCC维度，默认使用初始化时的值
            
        Returns:
            MFCC特征矩阵 (时间步, 特征维度)
        """
        y, sr = librosa.load(audio_path, sr=self.sr)
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=n_mfcc or self.n_mfcc
        )
        return mfcc.T

    def calculate_dtw_distance(self, student_mfcc: np.ndarray, 
                               reference_mfcc: np.ndarray) -> float:
        """
        计算DTW距离
        
        Args:
            student_mfcc: 学生发音的MFCC特征
            reference_mfcc: 标准发音的MFCC特征
            
        Returns:
            DTW距离值
        """
        distance, path = fastdtw(student_mfcc, reference_mfcc, dist=euclidean)
        return distance

    def pronunciation_score(self, student_audio: str, 
                           reference_audio: str) -> Dict:
        """
        计算发音评分
        
        Args:
            student_audio: 学生录音路径
            reference_audio: 标准发音路径
            
        Returns:
            包含评分和详细信息的字典
        """
        try:
            student_mfcc = self.extract_mfcc(student_audio)
            reference_mfcc = self.extract_mfcc(reference_audio)

            distance = self.calculate_dtw_distance(student_mfcc, reference_mfcc)

            score = max(0, 100 - (distance / self.max_distance) * 100)
            score = round(score, 1)

            return {
                "score": score,
                "distance": distance,
                "max_distance": self.max_distance,
                "level": self._get_level(score),
                "feedback": self._generate_feedback(score),
            }

        except Exception as e:
            raise RuntimeError(f"发音评价失败: {e}")

    def batch_pronunciation_score(self, pairs: List[Dict[str, str]]) -> List[Dict]:
        """
        批量计算发音评分
        
        Args:
            pairs: 列表，每个元素包含 {'student': 学生音频, 'reference': 标准音频}
            
        Returns:
            评分结果列表
        """
        results = []
        for pair in pairs:
            try:
                result = self.pronunciation_score(
                    pair['student'], 
                    pair['reference']
                )
                result['student_audio'] = pair['student']
                result['reference_audio'] = pair['reference']
                results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "student_audio": pair['student'],
                    "reference_audio": pair['reference'],
                })
        return results

    def _get_level(self, score: float) -> str:
        """根据评分获取等级"""
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

    def _generate_feedback(self, score: float) -> str:
        """生成反馈建议"""
        if score >= 90:
            return "发音非常标准，继续保持！"
        elif score >= 75:
            return "发音良好，注意个别音节的准确性"
        elif score >= 60:
            return "发音基本正确，需要加强练习"
        elif score >= 40:
            return "发音有待改进，建议多听标准发音"
        else:
            return "发音需要大幅改进，建议从基础音标开始练习"

    def compare_word_level(self, student_audio: str, 
                          reference_audio: str,
                          word_timestamps_ref: List[Dict]) -> List[Dict]:
        """
        单词级别的发音对比（需要参考音频的时间戳信息）
        
        Args:
            student_audio: 学生录音
            reference_audio: 标准录音
            word_timestamps_ref: 参考音频的单词时间戳列表
            
        Returns:
            每个单词的评分详情
        """
        student_y, _ = librosa.load(student_audio, sr=self.sr)
        reference_y, _ = librosa.load(reference_audio, sr=self.sr)

        word_scores = []
        for word_info in word_timestamps_ref:
            start = word_info['start']
            end = word_info['end']
            word = word_info['word']

            start_sample = int(start * self.sr)
            end_sample = int(end * self.sr)

            ref_segment = reference_y[start_sample:end_sample]
            
            if len(ref_segment) > 0:
                ref_mfcc = librosa.feature.mfcc(
                    y=ref_segment, 
                    sr=self.sr, 
                    n_mfcc=self.n_mfcc
                ).T
                
                student_segment = student_y[start_sample:end_sample]
                if len(student_segment) > 0:
                    student_mfcc = librosa.feature.mfcc(
                        y=student_segment, 
                        sr=self.sr, 
                        n_mfcc=self.n_mfcc
                    ).T
                    
                    distance = self.calculate_dtw_distance(student_mfcc, ref_mfcc)
                    score = max(0, 100 - (distance / self.max_distance) * 100)
                    
                    word_scores.append({
                        "word": word,
                        "score": round(score, 1),
                        "start": start,
                        "end": end,
                    })

        return word_scores