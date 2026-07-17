import math


class SmartFaceCrop:
    def __init__(self, target_w: int = 1080, target_h: int = 1920,
                 smoothing: float = 0.97, max_step: int = 8,
                 multi_speaker: bool = False, min_switch_interval: float = 1.5):
        self.target_w  = target_w
        self.target_h  = target_h
        self.smoothing = smoothing
        self.max_step  = max_step
        self.multi_speaker      = multi_speaker
        self.min_switch_interval = min_switch_interval

    def make_transform(self, clip):
        if self.multi_speaker:
            try:
                return self._make_multi_speaker_transform(clip)
            except ImportError:
                pass  # mediapipe недоступен — используем обычный однолицевой кроп
        return self._make_single_face_transform(clip)

    # ── Единственное "лучшее" лицо (без учёта, кто говорит) ────

    def _make_single_face_transform(self, clip):
        import cv2
        use_mp = False
        mp_det = None
        face_cascade = None
        try:
            import mediapipe as mp
            mp_det = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.45
            )
            use_mp = True
        except ImportError:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

        def _detect(frame_bgr):
            if use_mp:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                res = mp_det.process(rgb)
                if res.detections:
                    best = max(res.detections,
                               key=lambda d: d.score[0] if d.score else 0)
                    bb   = best.location_data.relative_bounding_box
                    fh, fw = frame_bgr.shape[:2]
                    return (int((bb.xmin + bb.width  / 2) * fw),
                            int((bb.ymin + bb.height / 2) * fh))
            else:
                gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
                if len(faces):
                    x, y, w, h = sorted(faces, key=lambda f2: f2[2] * f2[3])[-1]
                    return int(x + w / 2), int(y + h / 2)
            return None

        return clip.transform(self._smoothed_crop_frame(_detect))

    # ── Несколько лиц: кроп переключается на того, кто активно говорит ──

    def _make_multi_speaker_transform(self, clip):
        import cv2
        import mediapipe as mp

        MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT = 13, 14, 61, 291
        mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=2, refine_landmarks=False,
            min_detection_confidence=0.4, min_tracking_confidence=0.4,
        )

        tracks = []  # [{'cx','cy','mar_hist':[...], 'last_seen': t}, ...]
        active = {'idx': None, 'last_switch': -999.0}

        def _faces(frame_bgr):
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)
            out = []
            if res.multi_face_landmarks:
                fh, fw = frame_bgr.shape[:2]
                for lm in res.multi_face_landmarks:
                    pts = lm.landmark
                    xs = [p.x for p in pts]; ys = [p.y for p in pts]
                    cx = int((min(xs) + max(xs)) / 2 * fw)
                    cy = int((min(ys) + max(ys)) / 2 * fh)

                    def pt(i):
                        return (pts[i].x * fw, pts[i].y * fh)

                    mv = math.dist(pt(MOUTH_TOP), pt(MOUTH_BOTTOM))
                    mh = math.dist(pt(MOUTH_LEFT), pt(MOUTH_RIGHT))
                    out.append({'cx': cx, 'cy': cy, 'mar': mv / (mh + 1e-6)})
            return out

        def _assign(faces, t):
            for tr in tracks:
                tr['matched'] = False
            for face in faces:
                best_i, best_d = None, float('inf')
                for i, tr in enumerate(tracks):
                    if tr['matched']:
                        continue
                    d = (face['cx'] - tr['cx']) ** 2 + (face['cy'] - tr['cy']) ** 2
                    if d < best_d:
                        best_d, best_i = d, i
                if best_i is not None and best_d < 300 ** 2:
                    tr = tracks[best_i]
                    tr['cx'], tr['cy'] = face['cx'], face['cy']
                    tr['mar_hist'].append(face['mar'])
                    if len(tr['mar_hist']) > 15:
                        tr['mar_hist'].pop(0)
                    tr['last_seen'] = t
                    tr['matched'] = True
                else:
                    tracks.append({'cx': face['cx'], 'cy': face['cy'],
                                   'mar_hist': [face['mar']], 'last_seen': t,
                                   'matched': True})
            tracks[:] = [tr for tr in tracks if t - tr['last_seen'] < 2.0]

        def _activity(tr):
            h = tr['mar_hist']
            return 0.0 if len(h) < 2 else _stdev(h)

        def _choose_active(t):
            if not tracks:
                return None
            if len(tracks) == 1:
                return 0
            scores = [_activity(tr) for tr in tracks]
            best_i = max(range(len(scores)), key=lambda i: scores[i])
            if active['idx'] is None:
                active['idx'], active['last_switch'] = best_i, t
            elif (best_i != active['idx']
                  and t - active['last_switch'] > self.min_switch_interval
                  and scores[best_i] > scores[min(active['idx'], len(scores) - 1)] * 1.3):
                active['idx'], active['last_switch'] = best_i, t
            return min(active['idx'], len(tracks) - 1)

        def _detect(frame_bgr, t=0.0):
            faces = _faces(frame_bgr)
            if faces:
                _assign(faces, t)
            idx = _choose_active(t)
            if idx is not None and tracks:
                return tracks[idx]['cx'], tracks[idx]['cy']
            return None

        return clip.transform(self._smoothed_crop_frame(_detect, needs_time=True))

    # ── Общая логика сглаживания позиции и обрезки кадра ───────

    def _smoothed_crop_frame(self, detect_fn, needs_time: bool = False):
        import cv2
        tw, th = self.target_w, self.target_h
        alpha  = 1.0 - self.smoothing
        state  = {'cx': None, 'cy': None, 'raw_cx': None, 'raw_cy': None}

        def frame_fn(get_frame, t):
            frame_rgb = get_frame(t)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            fh, fw    = frame_bgr.shape[:2]

            face = detect_fn(frame_bgr, t) if needs_time else detect_fn(frame_bgr)
            if face:
                state['raw_cx'], state['raw_cy'] = face
            if state['raw_cx'] is None:
                state['raw_cx'] = fw // 2
                state['raw_cy'] = fh // 2

            raw_cx, raw_cy = state['raw_cx'], state['raw_cy']

            if state['cx'] is None:
                state['cx'], state['cy'] = float(raw_cx), float(raw_cy)
            else:
                tcx = state['cx'] * self.smoothing + raw_cx * alpha
                tcy = state['cy'] * self.smoothing + raw_cy * alpha
                dx, dy = tcx - state['cx'], tcy - state['cy']
                dist   = math.sqrt(dx * dx + dy * dy)
                if dist > self.max_step:
                    scale = self.max_step / dist
                    tcx   = state['cx'] + dx * scale
                    tcy   = state['cy'] + dy * scale
                state['cx'], state['cy'] = tcx, tcy

            cx, cy = int(state['cx']), int(state['cy'])
            x1 = max(0, min(fw - tw, cx - tw // 2))
            y1 = max(0, min(fh - th, cy - int(th * 0.33)))

            cropped = frame_bgr[y1:y1 + th, x1:x1 + tw]
            if cropped.shape[0] != th or cropped.shape[1] != tw:
                cropped = cv2.resize(cropped, (tw, th))
            return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        return frame_fn


def _stdev(values):
    n = len(values)
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / n) ** 0.5
