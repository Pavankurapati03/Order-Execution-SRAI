import os
from django.conf import settings

def resolve_media_path(path: str) -> str:
    """
    Convert /media/... URL to filesystem path safely
    """

    if isinstance(path, str) and path.startswith(settings.MEDIA_URL):
        # Remove MEDIA_URL and join properly
        relative_path = path[len(settings.MEDIA_URL):].lstrip("/\\")
        return os.path.normpath(
            os.path.join(str(settings.MEDIA_ROOT), relative_path)
        )

    return os.path.normpath(str(path))
