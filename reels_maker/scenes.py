def detect_scenes(video_path):
    try:
        from scenedetect import detect
        from scenedetect.detectors import ContentDetector
        scene_list = detect(video_path, ContentDetector())
        scenes = []
        for start_frame, end_frame in scene_list:
            s = start_frame.get_seconds()
            e = end_frame.get_seconds()
            if e - s >= 3:
                scenes.append((round(s, 2), round(e, 2)))
        return scenes
    except Exception:
        return []
