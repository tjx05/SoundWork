import os
import torchaudio

from models.whisper_asr import WhisperASR
# from models.diarization import WespeakerDiarizer
from recognition.speaker_reco import SpeakerRecognizer
from config import config

class MeetingDiary:
    def __init__(self):
        print("初始化...")
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
        # 说话人分割
        # print("分割中……")
        # segments=self.diarizer.diarize(audio_path)

        # 语音转文字
        print("转写中……")
        transcripts=self.asr.transcribe(audio_path)

        # 对齐+说话人识别
        print("对齐中……")
        results=[]
        unknown_counter=0

        for seg in transcripts:
            start=seg['start']
            end=seg['end']
            text=seg['text']

            # 提取片段
            temp_path=self.extract_segment(audio_path,start,end)

            # 识别说话人
            name,score=self.recognizer.identify(temp_path)

            if name is None:
                unknown_counter+=1
                speaker=f"SPEAKER_{unknown_counter:02d}"
                print(f"  [{start:.1f}s-{end:.1f}s] 未注册 -> {speaker} (相似度 {score:.4f})")
            else:
                speaker=name
                print(f"  [{start:.1f}s-{end:.1f}s] {speaker} (相似度 {score:.4f})")
            
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            results.append({
                "start":start,
                "end":end,
                "speaker":speaker,
                "emotion":"neutral",
                "text":text.strip()
            })
        
        # 合并相邻的统一说话人
        merged=self.merge_speaker(results)

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

# if __name__=="__main__":
#     diary=MeetingDiary()
#     results=diary.process("Data/mix/tem2.WAV")
#     diary.print_diary(results)

