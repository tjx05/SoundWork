import os
import torchaudio

from models.whisper_asr import WhisperASR
# from models.diarization import WespeakerDiarizer
from recognition.speaker_reco import SpeakerRecognizer
from recognition.emotion_compensated_reco import EmotionCompensatedRecognizer
from recognition.wav2vec2_reco import Wav2vec2Recognizer
from config import config

class MeetingDiary:
    def __init__(self,emotion_recognizer=None,use_compensation=True):
        print("初始化...")

        # 加载情感识别器
        self.emotion_recognizer=emotion_recognizer
        self.use_compensation=use_compensation

        if self.use_compensation:
            # 加载情感补偿的说话人识别
            self.recognizer=EmotionCompensatedRecognizer(
                model_path="speaker_checkpoints/best_model.pth",
                threshold=0.52,
                db_path="speaker_checkpoints/speaker_db",
                emotion_bias_path="speaker_checkpoints/emotion_bias.pth",
                emotion_recognizer=self.emotion_recognizer,
                compensation_strength=0.7,  # 最佳强度
                use_compensation=True
            )
        else:
            self.recognizer=SpeakerRecognizer(
                model_path="speaker_checkpoints/best_model.pth",
                threshold=0.52
            )
        # self.diarizer=WespeakerDiarizer(
        #     window_dur=1.2,
        #     step_dur=0.5,
        #     threshold=0.65,
        #     recognizer=self.recognizer
        # )
        self.asr=WhisperASR("base")
        self.sr=config.sr
        # # 把大模型存入实例中
        # self.emotion_recognizer = emotion_recognizer
    
    def extract_segment(self,audio_path,start,end):
        """提取音频片段并保存为临时文件"""
        signal,sr=torchaudio.load(audio_path)
        if sr!=self.sr:
            resampler=torchaudio.transforms.Resample(sr,self.sr)
            signal=resampler(signal)

        start_sample=int(start*self.sr)
        end_sample=int(end*self.sr)
        segment=signal[:,start_sample:end_sample]

        temp_path=f"temp/temp_segment.wav"
        torchaudio.save(temp_path,segment,self.sr)
        return temp_path
    
    def merge_speaker(self,results,gap_threshold=1):
        """合并相同说话人的文本"""
        if not results:
            return []
        
        merged=[]
        current=results[0].copy()
        for i in range(1,len(results)):
            r=results[i]
            if r['speaker']==current['speaker'] and (r['start']-current['end'])<gap_threshold:
                # 同一人，间隔小于阈值 → 合并
                current['end']=r['end']
                current['text']+=" "+r['text']
            else:
                merged.append(current)
                current=r.copy()

        merged.append(current)
        return merged
                
    
    def process(self,audio_path):
        # 语音转文字
        print("转写中……")
        transcripts=self.asr.transcribe(audio_path)

        # 对齐+说话人识别（支持自动注册和移动平均）
        print("对齐中……")
        results=[]

        for seg in transcripts:
            start=seg['start']
            end=seg['end']
            text=seg['text']
            duration=end-start  # 计算片段时长，用于质量评估

            # 提取片段
            temp_path=self.extract_segment(audio_path,start,end)

            # 识别说话人（自动注册陌生人，支持移动平均更新）
            speaker,score,source,gender,age=self.recognizer.identify_or_register(
                temp_path,segment_duration=duration
            )

            # 打印详细信息
            if source == "new_temp":
                print(f"  [{start:.1f}s-{end:.1f}s] 新说话人 {speaker} (自动注册, 时长{duration:.1f}s)")
            elif source == "temp":
                data = self.recognizer.temp_speakers.get(speaker, {})
                print(f"  [{start:.1f}s-{end:.1f}s] {speaker} (第{data.get('count', 0)}段, 相似度{score:.3f})")
            else:
                print(f"  [{start:.1f}s-{end:.1f}s] {speaker} (相似度{score:.3f})")

            if self.use_compensation:
                emotion_result=self.recognizer.get_last_emotion()
                emo=emotion_result.get('emotion', '中性')
                if gender=="未知":
                    gender=emotion_result.get('gender','未知')
                if age=="未知":
                    age=emotion_result.get('age','未知')

                # 如果是临时说话人，同步更新 recognizer 中的数据
                if source in ["new_temp", "temp"] and speaker in self.recognizer.temp_speakers:
                    if gender != "未知" and self.recognizer.temp_speakers[speaker].get("gender") == "未知":
                        self.recognizer.temp_speakers[speaker]["gender"] = gender

                    if age != "未知" and self.recognizer.temp_speakers[speaker].get("age") == "未知":
                        self.recognizer.temp_speakers[speaker]["age"] = age

            else:
                # 情感、性别、年龄识别
                emo="neutral"
                if self.emotion_recognizer:
                    try:
                        ai_res = self.emotion_recognizer.predict(temp_path)
                        emo = ai_res.get("emotion", "neutral")

                        # 只有当前性别/年龄是"未知"时，才用情感识别器的值覆盖
                        if source in ["new_temp", "temp"]:
                            updated = False
                            if gender == "未知":
                                new_gender = ai_res.get("gender", "未知")
                                if new_gender != "未知":
                                    gender = new_gender
                                    updated = True
                            if age == "未知":
                                new_age = ai_res.get("age", "未知")
                                if new_age != "未知":
                                    age = new_age
                                    updated = True
                            
                            # 同步更新 recognizer 中的临时说话人数据
                            if updated and speaker in self.recognizer.temp_speakers:
                                self.recognizer.temp_speakers[speaker]["gender"] = gender
                                self.recognizer.temp_speakers[speaker]["age"] = age

                    except Exception as e:
                        print(f"大模型识别失败: {e}")

            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

            results.append({
                "start":start,
                "end":end,
                "speaker":speaker,
                "emotion":emo,
                "gender":gender,
                "age":age,
                "text":text.strip()
            })

        # 合并相邻的统一说话人
        merged = self.merge_speaker(results)

        return merged

    
    # def align_reco(self,segments,transcripts,audio_path):
    #     """对齐+说话人识别"""
    #     results=[]
    #     used=set()  # 记录已经使用的文本,避免重复使用

    #     for seg in segments:
    #         # 找到落在该时间段内的文本
    #         text_parts=[]
    #         for i,t in enumerate(transcripts):
    #             if i in used:
    #                 continue
    #             if not (t['end'] <= seg['start'] or t['start'] >= seg['end']):
    #                 text_parts.append(t['text'])
    #                 used.add(i)

    #         # 空的就不加入结果
    #         if not text_parts:
    #             continue  

    #         results.append({
    #             "start":seg['start'],
    #             "end":seg['end'],
    #             "speaker":seg['speaker'],
    #             "text":" ".join(text_parts)
    #         })
    #     return results
    
    def print_diary(self,results):
        print("\n" + "="*50)
        print("会议日记")
        print("="*50)
        for r in results:
            print(f"[{r['start']:.1f}s -> {r['end']:.1f}s] {r['speaker']}: {r['text']}")

if __name__=="__main__":
    diary=MeetingDiary()
    results=diary.process("Data/mix/tem2.WAV")
    diary.print_diary(results)

