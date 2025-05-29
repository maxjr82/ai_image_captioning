from pathlib import Path
from typing import Optional, Dict, List
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import requests
import logging

logger = logging.getLogger(__name__)


class GooglePhotosClient:
    """Working implementation of Google Photos API client"""

    def __init__(self, credentials_path: str = "credentials.json"):
        self.credentials_path = Path(credentials_path)
        self.credentials = None
        self.service = self._authenticate()

    def _authenticate(self):
        """Handle OAuth2 authentication"""
        scopes = ["https://www.googleapis.com/auth/photoslibrary.readonly"]
        creds = None
        token_file = self.credentials_path.parent / "token.json"

        if token_file.exists():
            creds = Credentials.from_authorized_user_file(token_file, scopes)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), scopes
                )
                creds = flow.run_local_server(port=0)

            token_file.write_text(creds.to_json())

        self.credentials = creds
        return build(
            "photoslibrary", "v1", credentials=creds, static_discovery=False
        )

    def get_access_token(self) -> str:
        """
        Retrieves a valid access token for authenticated requests.
        """
        if not self.credentials.valid:
            if self.credentials.expired and self.credentials.refresh_token:
                self.credentials.refresh(Request())
            else:
                raise RuntimeError(
                    "Credentials are invalid and cannot be refreshed."
                )
        return self.credentials.token

    def _list_media_items(
        self, page_size: int = 100, page_token: Optional[str] = None
    ):
        """Correct implementation of mediaItems.list"""
        return (
            self.service.mediaItems()
            .list(pageSize=page_size, pageToken=page_token)
            .execute()
        )

    def _search_media_items(
        self, body: Dict, page_token: Optional[str] = None
    ):
        """Correct implementation of mediaItems.search"""
        if page_token:
            body["pageToken"] = page_token
        return self.service.mediaItems().search(body=body).execute()

    def get_photos(self, include_albums: bool = False) -> List[Dict]:
        """Get all photos with proper pagination handling"""
        photos = []
        next_page_token = None

        # First get all media items
        while True:
            try:
                response = self._list_media_items(page_token=next_page_token)
                items = response.get("mediaItems", [])
                photos.extend(
                    [
                        item
                        for item in items
                        if item["mimeType"].startswith("image/")
                    ]
                )
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
            except Exception as e:
                logger.error("Failed to list media items: %s", str(e))
                break

        if include_albums:
            return photos

        # Then get photos in albums
        album_photos = set()
        next_album_page_token = None

        while True:
            try:
                albums = (
                    self.service.albums()
                    .list(pageSize=50, pageToken=next_album_page_token)
                    .execute()
                )

                for album in albums.get("albums", []):
                    next_search_page_token = None
                    while True:
                        search_results = self._search_media_items(
                            body={"albumId": album["id"], "pageSize": 100},
                            page_token=next_search_page_token,
                        )
                        album_photos.update(
                            item["id"]
                            for item in search_results.get("mediaItems", [])
                        )
                        next_search_page_token = search_results.get(
                            "nextPageToken"
                        )
                        if not next_search_page_token:
                            break

                next_album_page_token = albums.get("nextPageToken")
                if not next_album_page_token:
                    break

            except Exception as e:
                logger.error("Failed to process albums: %s", str(e))
                break

        return [p for p in photos if p["id"] not in album_photos]

    def download_photo(
        self, photo_item: Dict, output_dir: Path
    ) -> Optional[Path]:
        """Download photo to local directory"""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            download_url = f"{photo_item['baseUrl']}=d"
            response = requests.get(download_url, timeout=30)
            response.raise_for_status()

            file_path = output_dir / photo_item["filename"]
            with open(file_path, "wb") as f:
                f.write(response.content)

            return file_path
        except Exception as e:
            logger.error(
                "Download failed for %s: %s",
                photo_item.get("filename"),
                str(e),
            )
            return None
