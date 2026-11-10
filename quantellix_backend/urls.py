import os
from django.urls import path, re_path, include
from django.conf import settings
from django.views.static import serve

urlpatterns = [
    # Include all portal routes
    path("", include("portal.urls")),

    # Static & Uploads file serving
    re_path(r"^portal/uploads/(?P<path>.*)$", serve, {"document_root": os.path.join(settings.BASE_DIR, "portal", "uploads")}),
    re_path(r"^uploads/(?P<path>.*)$", serve, {"document_root": os.path.join(settings.BASE_DIR, "portal", "uploads")}),
    re_path(r"^static/(?P<path>.*)$", serve, {"document_root": os.path.join(settings.BASE_DIR, "portal", "static")}),
]
