import numpy as np

from .analysis import MultimodalAnalyzer


class ViralityScorer:
    WEIGHTS = {
        'duration_score': 0.10, 'face_score':  0.20,
        'audio_energy':   0.15, 'speech_rate': 0.10,
        'sentiment_abs':  0.15, 'hook_score':  0.20,
        'laugh_bonus':    0.10,
    }

    @staticmethod
    def _duration_score(d: float) -> float:
        if 25 <= d <= 55: return 1.0
        return d / 25 if d < 25 else max(0.0, 1.0 - (d - 55) / 35)

    @staticmethod
    def _normalize(v, vmin, vmax):
        return 0.5 if vmax <= vmin else min(1.0, max(0.0, (v - vmin) / (vmax - vmin)))

    @classmethod
    def score_highlights(cls, highlights, words, audio_energies,
                         face_timeline, speech_rates, laugh_events):
        if not highlights:
            return highlights
        for h in highlights:
            st, et  = h.get('start_time', 0), h.get('end_time', 30)
            dur     = et - st
            snippet = h.get('_text_snippet', '')
            face_v  = [v for t, v in face_timeline.items() if st <= t <= et]
            eng_v   = [v for t, v in audio_energies.items() if st <= t <= et]
            sr_v    = [v for t, v in speech_rates.items()   if st <= t <= et]
            h['_features'] = {
                'duration':         dur,
                'duration_score':   cls._duration_score(dur),
                'face_score':       min(1.0, np.mean(face_v)) if face_v else 0.3,
                'audio_energy_raw': float(np.mean(eng_v)) if eng_v else 0.0,
                'speech_rate_raw':  float(np.mean(sr_v))  if sr_v  else 0.0,
                'sentiment_abs':    abs(MultimodalAnalyzer.sentiment_score(snippet)),
                'hook_score':       MultimodalAnalyzer.hook_score(words, st),
                'laugh_bonus':      1.0 if any(st <= e <= et for e in laugh_events) else 0.0,
            }
        all_ae = [h['_features']['audio_energy_raw'] for h in highlights]
        all_sr = [h['_features']['speech_rate_raw']  for h in highlights]
        ae_min, ae_max = min(all_ae), max(all_ae)
        sr_min, sr_max = min(all_sr), max(all_sr)
        for h in highlights:
            f = h['_features']
            score = (
                cls.WEIGHTS['duration_score'] * f['duration_score'] +
                cls.WEIGHTS['face_score']     * f['face_score'] +
                cls.WEIGHTS['audio_energy']   * cls._normalize(f['audio_energy_raw'], ae_min, ae_max) +
                cls.WEIGHTS['speech_rate']    * cls._normalize(f['speech_rate_raw'],  sr_min, sr_max) +
                cls.WEIGHTS['sentiment_abs']  * f['sentiment_abs'] +
                cls.WEIGHTS['hook_score']     * f['hook_score'] +
                cls.WEIGHTS['laugh_bonus']    * f['laugh_bonus']
            )
            h['virality_score'] = round(score, 4)
        highlights.sort(key=lambda x: x['virality_score'], reverse=True)
        return highlights
