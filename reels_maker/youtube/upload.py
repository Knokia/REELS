import datetime
import json
import os

from ..config import QUOTA_PATH
from .auth import ensure_fresh

# Стоимости в units согласно YouTube Data API v3 (quota costs документация Google)
QUOTA_COST_VIDEO_INSERT     = 1600
QUOTA_COST_THUMBNAIL_SET    = 50
QUOTA_COST_PLAYLIST_INSERT  = 50
DEFAULT_DAILY_QUOTA         = 10000


def _today() -> str:
    return datetime.date.today().isoformat()


def _load_quota_state() -> dict:
    if not os.path.exists(QUOTA_PATH):
        return {"date": _today(), "used": 0}
    try:
        with open(QUOTA_PATH, encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        return {"date": _today(), "used": 0}
    if state.get("date") != _today():
        return {"date": _today(), "used": 0}
    return state


def _record_usage(units: int) -> None:
    state = _load_quota_state()
    state["used"] = state.get("used", 0) + units
    with open(QUOTA_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f)


def quota_used_today() -> int:
    return _load_quota_state().get("used", 0)


def quota_remaining(daily_quota: int = DEFAULT_DAILY_QUOTA) -> int:
    return max(0, daily_quota - quota_used_today())


def get_channel_info(credentials) -> dict:
    import googleapiclient.discovery
    ensure_fresh(credentials)
    yt = googleapiclient.discovery.build("youtube", "v3", credentials=credentials)
    resp = yt.channels().list(part="snippet", mine=True).execute()
    items = resp.get("items", [])
    if not items:
        return {}
    return {"id": items[0]["id"], "title": items[0]["snippet"]["title"]}


def list_playlists(credentials) -> list:
    import googleapiclient.discovery
    ensure_fresh(credentials)
    yt = googleapiclient.discovery.build("youtube", "v3", credentials=credentials)
    playlists, page_token = [], None
    while True:
        resp = yt.playlists().list(
            part="snippet", mine=True, maxResults=50, pageToken=page_token
        ).execute()
        for item in resp.get("items", []):
            playlists.append({"id": item["id"], "title": item["snippet"]["title"]})
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return playlists


def upload_video(credentials, clip_path, title, description, tags, category,
                  privacy, publish_at=None, thumbnail_path=None, playlist_id=None,
                  progress_cb=None) -> dict:
    """Uploads a video (+ optional thumbnail/playlist assignment). Returns {'video_id', 'url'}."""
    import googleapiclient.discovery
    import googleapiclient.http

    ensure_fresh(credentials)
    yt = googleapiclient.discovery.build("youtube", "v3", credentials=credentials)

    body = {
        "snippet": {"title": title, "description": description,
                    "tags": tags, "categoryId": category},
        "status":  {"privacyStatus": privacy, "selfDeclaredMadeForKids": False},
    }
    if publish_at:
        body["status"]["publishAt"] = publish_at

    media = googleapiclient.http.MediaFileUpload(
        clip_path, mimetype="video/mp4", resumable=True, chunksize=5 * 1024 * 1024
    )
    req = yt.videos().insert(part=",".join(body.keys()), body=body, media_body=media)
    response = None
    while response is None:
        status, response = req.next_chunk()
        if progress_cb and status:
            progress_cb(status.progress())
    _record_usage(QUOTA_COST_VIDEO_INSERT)

    video_id = response.get("id", "")

    if thumbnail_path and os.path.exists(thumbnail_path):
        yt.thumbnails().set(
            videoId=video_id,
            media_body=googleapiclient.http.MediaFileUpload(thumbnail_path)
        ).execute()
        _record_usage(QUOTA_COST_THUMBNAIL_SET)

    if playlist_id:
        yt.playlistItems().insert(
            part="snippet",
            body={"snippet": {"playlistId": playlist_id,
                               "resourceId": {"kind": "youtube#video", "videoId": video_id}}}
        ).execute()
        _record_usage(QUOTA_COST_PLAYLIST_INSERT)

    return {"video_id": video_id, "url": f"https://www.youtube.com/watch?v={video_id}"}
