from __future__ import annotations

from collections import Counter
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from solver import is_solvable_state
from vision import COLOR_KEYS, as_debug_dict, detect_face_from_image_bytes

app = FastAPI(
    title="Cube Scanner API",
    description="Capture cube faces with OpenCV and validate state for Kociemba solver.",
)

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

FACE_ORDER = ["U", "R", "F", "D", "L", "B"]
FACE_INSTRUCTION = {
    "U": "Show face with WHITE center",
    "R": "Show face with RED center",
    "F": "Show face with GREEN center",
    "D": "Show face with YELLOW center",
    "L": "Show face with ORANGE center",
    "B": "Show face with BLUE center",
}
CENTER_EXPECTED = {
    "U": "W",
    "R": "R",
    "F": "G",
    "D": "Y",
    "L": "O",
    "B": "B",
}

COLOR_TO_FACE = {
    "W": "U",
    "R": "R",
    "G": "F",
    "Y": "D",
    "O": "L",
    "B": "B",
}

# Allow frontend to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Since it's a local project, allow everything
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


class FinalizeStateRequest(BaseModel):
    faces: dict[str, list[str]] = Field(
        description="Detected faces keyed by U,R,F,D,L,B; each value is 9 color chars [W,R,G,Y,O,B]."
    )


@app.get("/")
def index() -> FileResponse:
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    return FileResponse(str(index_file))


@app.get("/api/capture-order")
def get_capture_order():
    return {
        "order": FACE_ORDER,
        "instructions": [FACE_INSTRUCTION[f] for f in FACE_ORDER],
        "expected_centers": [CENTER_EXPECTED[f] for f in FACE_ORDER],
    }


@app.post("/api/detect-face")
async def detect_face(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        detected = detect_face_from_image_bytes(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return as_debug_dict(detected)


def _validate_face_payload(faces: dict[str, list[str]]) -> tuple[bool, str | None]:
    for face in FACE_ORDER:
        if face not in faces:
            return False, f"Missing face: {face}"
        if len(faces[face]) != 9:
            return False, f"Face {face} must have exactly 9 stickers"
        invalid = [c for c in faces[face] if c not in COLOR_KEYS]
        if invalid:
            return False, f"Face {face} has invalid colors: {invalid}"
        center = faces[face][4]
        if center != CENTER_EXPECTED[face]:
            return (
                False,
                f"Face {face} center expected {CENTER_EXPECTED[face]}, but got {center}. Please recapture this face.",
            )
    return True, None


def _build_kociemba_state(faces: dict[str, list[str]]) -> str:
    colors = []
    for face in FACE_ORDER:
        colors.extend(faces[face])
    return "".join(COLOR_TO_FACE[c] for c in colors)


@app.post("/api/finalize-state")
def finalize_state(payload: FinalizeStateRequest):
    ok, error = _validate_face_payload(payload.faces)
    if not ok:
        raise HTTPException(status_code=400, detail=error)

    all_colors = [c for f in FACE_ORDER for c in payload.faces[f]]
    counts = Counter(all_colors)
    for color in COLOR_KEYS:
        if counts[color] != 9:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid color count: {color} appears {counts[color]} times (must be 9)",
            )

    state = _build_kociemba_state(payload.faces)
    solvable, solution, solve_error = is_solvable_state(state)

    return {
        "cube_state": state,
        "is_valid": True,
        "is_solvable": solvable,
        "solution": solution,
        "error": solve_error,
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
