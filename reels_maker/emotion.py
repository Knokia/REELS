import math
import os

import numpy as np


class EmotionDetector:
    """
    Определяет эмоцию по геометрии лица через MediaPipe Face Mesh.
    Не требует DeepFace / FER / hsemotion — только mediapipe + opencv.

    Признаки:
        MAR  — Mouth Aspect Ratio  (открытость рта)
        EAR  — Eye Aspect Ratio    (открытость глаз)
        BROW — относительная высота бровей над глазами
        HEAD_TILT — наклон головы

    Правила (детерминированные пороги):
        surprise  : MAR > 0.35  AND BROW > 0.20
        happy     : MAR > 0.22  AND BROW 0.13-0.20
        fear      : MAR > 0.20  AND BROW > 0.22  AND EAR > 0.28
        angry     : BROW < 0.10 AND MAR < 0.15
        sad       : BROW 0.10-0.13 AND HEAD_TILT > 5°
        disgust   : BROW < 0.12 AND MAR > 0.12  AND MAR < 0.22
        neutral   : всё остальное
    """

    # Индексы MediaPipe Face Mesh (468 точек)
    # Рот
    _MOUTH_TOP    = 13
    _MOUTH_BOTTOM = 14
    _MOUTH_LEFT   = 61
    _MOUTH_RIGHT  = 291

    # Глаза (верх/низ/лево/право)
    _LEFT_EYE_TOP    = 159
    _LEFT_EYE_BOTTOM = 145
    _LEFT_EYE_LEFT   = 33
    _LEFT_EYE_RIGHT  = 133
    _RIGHT_EYE_TOP   = 386
    _RIGHT_EYE_BOTTOM= 374
    _RIGHT_EYE_LEFT  = 362
    _RIGHT_EYE_RIGHT = 263

    # Брови (центральные точки)
    _LEFT_BROW_TOP   = 105
    _RIGHT_BROW_TOP  = 334

    # Лицо (верх/низ)
    _FACE_TOP    = 10
    _FACE_BOTTOM = 152

    # Нос (для наклона)
    _NOSE_TIP  = 4
    _LEFT_EAR  = 234
    _RIGHT_EAR = 454

    @classmethod
    def _get_mesh(cls):
        """Синглтон MediaPipe FaceMesh."""
        if not hasattr(cls, '_mesh_instance') or cls._mesh_instance is None:
            try:
                import mediapipe as mp
                cls._mesh_instance = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.4,
                    min_tracking_confidence=0.4,
                )
            except Exception:
                cls._mesh_instance = None
        return cls._mesh_instance

    @classmethod
    def _landmarks(cls, frame_bgr):
        """
        Возвращает список landmark-объектов или None.
        """
        mesh = cls._get_mesh()
        if mesh is None:
            return None
        try:
            import cv2
            rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = mesh.process(rgb)
            if results.multi_face_landmarks:
                return results.multi_face_landmarks[0].landmark
        except Exception:
            pass
        return None

    @classmethod
    def _features(cls, lm, h: int, w: int) -> dict:
        """
        Вычисляет геометрические признаки из landmark-списка.
        """
        def pt(idx) -> np.ndarray:
            return np.array([lm[idx].x * w, lm[idx].y * h])

        # ── Mouth Aspect Ratio ──
        mouth_v = np.linalg.norm(pt(cls._MOUTH_TOP)    - pt(cls._MOUTH_BOTTOM))
        mouth_h = np.linalg.norm(pt(cls._MOUTH_LEFT)   - pt(cls._MOUTH_RIGHT))
        mar     = mouth_v / (mouth_h + 1e-6)

        # ── Eye Aspect Ratio (среднее двух глаз) ──
        def ear(top, bot, left, right):
            v = np.linalg.norm(pt(top) - pt(bot))
            h2 = np.linalg.norm(pt(left) - pt(right))
            return v / (h2 + 1e-6)

        ear_val = (ear(cls._LEFT_EYE_TOP,  cls._LEFT_EYE_BOTTOM,
                       cls._LEFT_EYE_LEFT, cls._LEFT_EYE_RIGHT) +
                   ear(cls._RIGHT_EYE_TOP, cls._RIGHT_EYE_BOTTOM,
                       cls._RIGHT_EYE_LEFT, cls._RIGHT_EYE_RIGHT)) / 2.0

        # ── Brow height (нормировано на высоту лица) ──
        face_h   = np.linalg.norm(pt(cls._FACE_TOP) - pt(cls._FACE_BOTTOM)) + 1e-6
        brow_y   = (pt(cls._LEFT_BROW_TOP)[1]  + pt(cls._RIGHT_BROW_TOP)[1])  / 2
        eye_y    = (pt(cls._LEFT_EYE_TOP)[1]   + pt(cls._RIGHT_EYE_TOP)[1])   / 2
        brow_rel = (eye_y - brow_y) / face_h   # больше = брови выше

        # ── Head tilt (крен головы в градусах) ──
        left_ear_pt  = pt(cls._LEFT_EAR)
        right_ear_pt = pt(cls._RIGHT_EAR)
        dx = right_ear_pt[0] - left_ear_pt[0]
        dy = right_ear_pt[1] - left_ear_pt[1]
        head_tilt = abs(math.degrees(math.atan2(dy, dx + 1e-6)))

        return {
            'mar':       mar,
            'ear':       ear_val,
            'brow':      brow_rel,
            'head_tilt': head_tilt,
        }

    @classmethod
    def _classify(cls, f: dict) -> str:
        """Детерминированная классификация по порогам."""
        mar  = f['mar']
        ear  = f['ear']
        brow = f['brow']
        tilt = f['head_tilt']

        if mar > 0.35 and brow > 0.20:
            return 'surprise'
        if mar > 0.20 and brow > 0.22 and ear > 0.28:
            return 'fear'
        if mar > 0.22 and 0.13 <= brow <= 0.22:
            return 'happy'
        if brow < 0.10 and mar < 0.15:
            return 'angry'
        if 0.10 <= brow <= 0.13 and tilt > 5:
            return 'sad'
        if brow < 0.12 and 0.12 < mar < 0.22:
            return 'disgust'
        return 'neutral'

    # ── Fallback: OpenCV Haar + эвристика только по рту ────

    @classmethod
    def _opencv_fallback(cls, frame_bgr) -> str:
        """
        Если MediaPipe недоступен — пробуем OpenCV LBP smile detector.
        """
        try:
            import cv2
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            # Smile cascade (поставляется вместе с opencv-python)
            smile_xml = os.path.join(
                os.path.dirname(cv2.__file__),
                'data', 'haarcascade_smile.xml'
            )
            # Альтернативный путь
            if not os.path.exists(smile_xml):
                smile_xml = cv2.data.haarcascades + 'haarcascade_smile.xml'

            face_xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade  = cv2.CascadeClassifier(face_xml)
            smile_cascade = cv2.CascadeClassifier(smile_xml)

            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
            if len(faces) == 0:
                return 'neutral'

            x, y, w, h = sorted(faces, key=lambda f2: f2[2]*f2[3])[-1]
            roi = gray[y:y+h, x:x+w]

            smiles = smile_cascade.detectMultiScale(
                roi, scaleFactor=1.7, minNeighbors=22,
                minSize=(25, 25)
            )
            return 'happy' if len(smiles) > 0 else 'neutral'
        except Exception:
            return 'neutral'

    # ── Публичный API ────────────────────────────────────────

    @classmethod
    def detect(cls, frame_bgr) -> str:
        """Определяет эмоцию одного кадра. Всегда возвращает строку."""
        lm = cls._landmarks(frame_bgr)
        if lm is not None:
            h, w = frame_bgr.shape[:2]
            try:
                features = cls._features(lm, h, w)
                return cls._classify(features)
            except Exception:
                pass
        # MediaPipe не сработал — OpenCV fallback
        return cls._opencv_fallback(frame_bgr)

    @classmethod
    def detect_timeline(cls, video_path: str, sample_fps: float = 0.5) -> dict:
        """
        Сэмплирует видео и возвращает {time_sec: emotion_str}.
        """
        result = {}
        try:
            import cv2
            cap   = cv2.VideoCapture(video_path)
            fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step  = max(1, int(fps / sample_fps))
            idx   = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % step == 0:
                    t = round(idx / fps, 2)
                    result[t] = cls.detect(frame)
                idx += 1
            cap.release()
        except Exception:
            pass
        return result

    @classmethod
    def cleanup(cls):
        """Освобождает ресурсы MediaPipe."""
        if hasattr(cls, '_mesh_instance') and cls._mesh_instance is not None:
            try:
                cls._mesh_instance.close()
            except Exception:
                pass
            cls._mesh_instance = None
