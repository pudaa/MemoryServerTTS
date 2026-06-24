"""Dashboard router —— 管理后台调试面板"""
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["Dashboard"])

_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "index.html"
_CACHED_HTML: str | None = None


def _get_html() -> str:
    global _CACHED_HTML
    if _CACHED_HTML is None:
        _CACHED_HTML = _TEMPLATE_PATH.read_text(encoding="utf-8")
    return _CACHED_HTML


@router.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    """管理后台 —— 集成所有模块的调试/测速面板"""
    return _get_html()


@router.get("/admin/", response_class=HTMLResponse)
async def admin_dashboard_slash():
    return _get_html()
