import os
import tempfile

import numpy as np
from moviepy import VideoFileClip


class MultimodalAnalyzer:

    @staticmethod
    def load_audio(video_path: str, sr: int = 22050):
        """Извлекает аудио из видео один раз (вместо того чтобы каждая из
        4 функций ниже отдельно гоняла ffmpeg+librosa по новой). Бросает
        исключение при неудаче — пусть вызывающий код явно залогирует причину,
        а не тихо получит нули по всем метрикам."""
        import librosa
        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(wav_path, logger=None)
            clip.close()
            return librosa.load(wav_path, sr=sr, mono=True)
        finally:
            if wav_path:
                try: os.unlink(wav_path)
                except Exception: pass

    @staticmethod
    def extract_audio_energy(y, sr, hop: float = 0.5) -> dict:
        try:
            hop_s    = int(hop * sr)
            energies = {}
            for i in range(0, len(y) - hop_s, hop_s):
                t   = i / sr
                rms = float(np.sqrt(np.mean(y[i:i + hop_s] ** 2)))
                energies[round(t, 2)] = rms
            return energies
        except Exception:
            return {}

    @staticmethod
    def detect_beats(y, sr) -> list:
        try:
            import librosa
            _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            return librosa.frames_to_time(beat_frames, sr=sr).tolist()
        except Exception:
            return []

    @staticmethod
    def detect_silence_gaps(y, sr, threshold_db: float = -40.0,
                            min_silence_sec: float = 0.3) -> list:
        try:
            import librosa
            hop    = 512
            rms    = librosa.feature.rms(y=y, hop_length=hop)[0]
            db     = librosa.amplitude_to_db(rms, ref=np.max)
            times  = librosa.frames_to_time(np.arange(len(db)), sr=sr, hop_length=hop)
            silent = []
            in_s   = False
            s_start= 0.0
            for t, d in zip(times, db):
                if d < threshold_db and not in_s:
                    in_s, s_start = True, t
                elif d >= threshold_db and in_s:
                    in_s = False
                    if t - s_start >= min_silence_sec:
                        silent.append((round(s_start, 3), round(t, 3)))
            return silent
        except Exception:
            return []

    @staticmethod
    def detect_laugh_applause(y, sr) -> list:
        try:
            import librosa
            hop    = 512
            sc     = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
            sb     = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)[0]
            rms    = librosa.feature.rms(y=y, hop_length=hop)[0]
            times  = librosa.frames_to_time(np.arange(len(sc)), sr=sr, hop_length=hop)
            sc_m, sc_s = np.mean(sc), np.std(sc)
            rms_m  = np.mean(rms)
            events = []
            for t, s, b, r in zip(times, sc, sb, rms):
                if s > sc_m + sc_s and b > 2000 and r > rms_m * 1.5:
                    events.append(round(float(t), 2))
            merged = []
            for e in events:
                if not merged or e - merged[-1] > 1.0:
                    merged.append(e)
            return merged
        except Exception:
            return []

    @staticmethod
    def speech_rate(words: list, window: float = 5.0) -> dict:
        if not words:
            return {}
        duration = words[-1]['end']
        result   = {}
        t = 0.0
        while t < duration:
            ws = [w for w in words if t <= w['start'] < t + window]
            result[round(t, 1)] = len(ws) / window
            t += window / 2
        return result

    @staticmethod
    def detect_faces_timeline(video_path: str, sample_fps: float = 1.0) -> dict:
        counts, _, _ = MultimodalAnalyzer.detect_visual_timeline(video_path, sample_fps)
        return counts

    @staticmethod
    def detect_visual_timeline(video_path: str, sample_fps: float = 1.0):
        """Один проход по видео -> три таймлайна:
          counts[t] — сколько лиц найдено в кадре;
          motion[t] — насколько кадр отличается от предыдущего (0..1), нужно, чтобы
                      отличать живое действие от статичной заставки/фотографии,
                      которую иначе скоринг спокойно берёт в топ на все 25 секунд;
          boxes[t]  — рамки лиц в НОРМАЛИЗОВАННЫХ координатах (x, y, w, h в долях
                      кадра), по ним режим «по центру» подрезает исходник к зоне
                      действия, чтобы люди не были крошечными в общем плане.
        Всё считается за один проход намеренно: декодирование часового видео стоит
        минуты, а раньше здесь уже был ровно такой же цикл ради одних лишь счётчиков."""
        try:
            import cv2
            import numpy as np
            counts, motion, boxes = {}, {}, {}
            cap    = cv2.VideoCapture(video_path)
            fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step   = max(1, int(fps / sample_fps))
            use_mp = False
            mp_det = None
            try:
                import mediapipe as mp
                mp_det = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5
                )
                use_mp = True
            except ImportError:
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            frame_idx = 0
            prev_small = None
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % step == 0:
                    t = round(frame_idx / fps, 2)
                    fh, fw = frame.shape[:2]
                    if use_mp:
                        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        res   = mp_det.process(rgb)
                        dets  = res.detections or []
                        count = len(dets)
                        bb = []
                        for d in dets:
                            r = d.location_data.relative_bounding_box
                            bb.append((r.xmin, r.ymin, r.width, r.height))
                    else:
                        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                        count = len(faces)
                        bb = [(x / fw, y / fh, w / fw, h / fh) for x, y, w, h in faces]
                    counts[t] = count
                    boxes[t]  = bb

                    # Дешёвая метрика движения: средняя разница яркости с предыдущим
                    # сэмплом на сильно уменьшенной копии кадра. Статичная картинка
                    # даёт ~0, живое действие/смена плана — заметно больше.
                    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36))
                    if prev_small is not None:
                        diff = np.abs(small.astype(np.int16) - prev_small.astype(np.int16))
                        motion[t] = float(diff.mean()) / 255.0
                    else:
                        motion[t] = 0.0
                    prev_small = small
                frame_idx += 1
            cap.release()
            if mp_det:
                mp_det.close()
            return counts, motion, boxes
        except Exception:
            return {}, {}, {}

    @staticmethod
    def sentiment_score(text: str) -> float:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            return SentimentIntensityAnalyzer().polarity_scores(text)['compound']
        except ImportError:
            pass
        try:
            from rusentiment import RuSentiment
            label = RuSentiment().predict([text])[0]
            return 1.0 if label == 'positive' else (-1.0 if label == 'negative' else 0.0)
        except ImportError:
            pass
        pos_w = {'отлично','супер','прекрасно','круто','невероятно',
                 'great','amazing','incredible','awesome','fantastic'}
        neg_w = {'плохо','ужасно','кошмар','провал','катастрофа',
                 'bad','terrible','awful','horrible','disaster'}
        ws    = set(text.lower().split())
        pos, neg = len(ws & pos_w), len(ws & neg_w)
        return (pos - neg) / (pos + neg) if pos + neg > 0 else 0.0

    @staticmethod
    def find_sentence_boundaries(words: list) -> list:
        boundaries = []
        for w in words:
            t = w['word'].strip()
            if any(t.endswith(e) for e in {'.','!','?','...','…'}) or t in {'.','!','?'}:
                boundaries.append(w['end'])
        return boundaries

    @staticmethod
    def snap_to_sentence_boundary(t: float, boundaries: list,
                                  tolerance: float = 3.0) -> float:
        if not boundaries:
            return t
        closest = min(boundaries, key=lambda b: abs(b - t))
        return closest if abs(closest - t) <= tolerance else t

    @staticmethod
    def snap_to_silence(t: float, silences: list, tolerance: float = 2.0) -> float:
        if not silences:
            return t
        candidates = [(s + e) / 2 for s, e in silences
                      if abs((s + e) / 2 - t) <= tolerance]
        return min(candidates, key=lambda c: abs(c - t)) if candidates else t

    @staticmethod
    def snap_to_beat(t: float, beats: list, tolerance: float = 0.5) -> float:
        if not beats:
            return t
        closest = min(beats, key=lambda b: abs(b - t))
        return closest if abs(closest - t) <= tolerance else t

    @staticmethod
    def hook_score(words: list, start: float, duration: float = 5.0) -> float:
        hook_words = {
            'почему','как','зачем','когда','что','кто','где',
            'why','how','what','who','when','where',
            'никогда','всегда','только','вдруг','never','always',
            'only','just','suddenly','секрет','тайна','шок',
            'secret','shocking','incredible','truth',
        }
        ws = [w for w in words if start <= w['start'] < start + duration]
        if not ws:
            return 0.0
        speed    = len(ws) / duration
        kw_count = sum(1 for w in ws if w['word'].lower().strip('.,!?') in hook_words)
        return round(min(1.0, speed / 3.0) * 0.5 + min(1.0, kw_count / 2.0) * 0.5, 3)
