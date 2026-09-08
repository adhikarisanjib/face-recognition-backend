import json
import random
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from deepface import DeepFace
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config.database import get_db
from app.config.settings import settings
from app.models import Face, Person, RecognitionResult
from app.schema import (
    FaceForm,
    FaceResponse,
    PersonForm,
    PersonResponse,
    RecognitionMediaResponse,
    RecognitionResponse,
)
from app.services.auth import get_authenticated_user
from app.utils import draw_face_result, find_best_face_match

router = APIRouter(tags=["face"])


@router.get("/persons", response_model=list[PersonResponse])
async def get_persons(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Person))
    persons = result.scalars().all()
    return persons


@router.post("/person", response_model=PersonResponse)
async def create_person(
    person_form: PersonForm,
    user=Depends(get_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    new_person = Person(name=person_form.name, user_id=person_form.user_id)
    db.add(new_person)
    await db.commit()
    await db.refresh(new_person)
    return new_person


@router.get("/person/{person_id}", response_model=PersonResponse)
async def get_person(person_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Person).where(Person.id == person_id))
    person = result.scalar_one_or_none()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    return person


@router.put("/person/{person_id}", response_model=PersonResponse)
async def update_person(
    person_id: int,
    person_form: PersonForm | None = None,
    user=Depends(get_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    result = await db.execute(select(Person).where(Person.id == person_id))
    person = result.scalar_one_or_none()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    if person_form is not None:
        if person_form.name is not None:
            person.name = person_form.name
        if person_form.user_id is not None:
            person.user_id = person_form.user_id

    await db.commit()
    await db.refresh(person)
    return person


@router.delete("/person/{person_id}", response_model=PersonResponse)
async def delete_person(
    person_id: int,
    user=Depends(get_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not user.role == "admin":
        raise HTTPException(
            status_code=403, detail="You do not have permission to delete persons."
        )

    result = await db.execute(select(Person).where(Person.id == person_id))
    person = result.scalar_one_or_none()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    await db.delete(person)
    await db.commit()
    return person


@router.get("/person/{person_id}/faces", response_model=list[FaceResponse])
async def get_faces_for_person(person_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Person).options(selectinload(Person.faces)).where(Person.id == person_id)
    )

    person = result.scalar_one_or_none()

    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    return person.faces


@router.get("/faces", response_model=list[FaceResponse])
async def get_faces(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Face))
    faces = result.scalars().all()
    return faces


@router.get("/face/{face_id}", response_model=FaceResponse)
async def get_face(face_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Face).where(Face.id == face_id))
    face = result.scalar_one_or_none()
    if not face:
        raise HTTPException(status_code=404, detail="Face not found")
    return face


@router.post("/face/{person_id}", response_model=FaceResponse)
async def create_face(
    request: Request,
    person_id: int,
    face_form: FaceForm = Depends(),
    user=Depends(get_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    result = await db.execute(select(Person).where(Person.id == person_id))
    person = result.scalar_one_or_none()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    try:
        image = cv2.imdecode(
            np.frombuffer(face_form.file.file.read(), np.uint8), cv2.IMREAD_COLOR
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detected_faces = []
    try:
        detected_faces = DeepFace.represent(
            img_path=image_rgb,
            model_name=settings.FACE_EMBEDDING_MODEL,
            enforce_detection=True,
            detector_backend="opencv",
        )
    except Exception as e:
        print(f"Error during face detection: {e}")
        raise HTTPException(status_code=400, detail="Error during face detection.")

    if len(detected_faces) > 1:
        raise HTTPException(
            status_code=400,
            detail="Multiple faces detected in the image. Please upload an image with a single face.",
        )
    elif len(detected_faces) == 0:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the image. Please upload an image with a clear visible face.",
        )

    first_face: dict[str, Any] = detected_faces[0]
    x, y, w, h = (
        first_face["facial_area"]["x"],
        first_face["facial_area"]["y"],
        first_face["facial_area"]["w"],
        first_face["facial_area"]["h"],
    )
    face_crop = image_rgb[y : y + h, x : x + w]

    face_filename = f"face_{uuid4().hex}.jpg"
    face_dir = settings.MEDIA_DIR / "faces"
    face_dir.mkdir(parents=True, exist_ok=True)
    face_path = settings.MEDIA_DIR / "faces" / face_filename

    cv2.imwrite(str(face_path), cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))

    face = Face(
        person_id=person.id,
        image_path=str(request.url_for("media", path=f"faces/{face_filename}")),
        embedding=first_face["embedding"],
    )

    db.add(face)
    await db.commit()
    await db.refresh(face)
    return face


@router.delete("/face/{face_id}", response_model=FaceResponse)
async def delete_face(
    face_id: int,
    user=Depends(get_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not user.role == "admin":
        raise HTTPException(
            status_code=403, detail="You do not have permission to delete faces."
        )

    result = await db.execute(select(Face).where(Face.id == face_id))
    face = result.scalar_one_or_none()
    if not face:
        raise HTTPException(status_code=404, detail="Face not found")

    await db.delete(face)
    await db.commit()
    return face


@router.post("/recognize/image", response_model=RecognitionMediaResponse)
async def recognize_faces_in_image(
    request: Request,
    face_form: FaceForm = Depends(),
    user=Depends(get_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    image_bytes = await face_form.file.read()

    image = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        detected_faces = await run_in_threadpool(
            DeepFace.represent,
            img_path=image,
            model_name=settings.FACE_EMBEDDING_MODEL,
            enforce_detection=True,
            detector_backend="opencv",
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Face not detected in the image")

    if not detected_faces:
        raise HTTPException(
            status_code=400,
            detail=(
                "No face detected in the image. "
                "Please upload an image with a clear visible face."
            ),
        )

    recognition_results = []

    for detected_face in detected_faces:
        embedding = detected_face["embedding"]
        facial_area = detected_face["facial_area"]

        face, distance = await find_best_face_match(embedding=embedding, db=db)

        is_match, confidence = draw_face_result(
            image=image, facial_area=facial_area, face=face, distance=distance
        )

        recognition_results.append(
            RecognitionResponse(
                person_id=(face.person.id if is_match else None),
                name=(face.person.name if is_match else "Unknown"),
                confidence=confidence,
                message=("Match found" if is_match else "No match found"),
            )
        )

    # Save annotated image
    filename = f"image_{uuid4().hex}.jpg"

    output_path = settings.MEDIA_DIR / "recognition"
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path / filename), image)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to save recognition image")

    media_url = str(request.url_for("media", path=f"recognition/{filename}"))

    # Save recognition results
    recognition_result = RecognitionResult(
        ran_by=user.id if user else None,
        media_path=str(output_path / filename),
        recognition_results=[result.model_dump() for result in recognition_results],
    )
    db.add(recognition_result)
    await db.commit()

    return RecognitionMediaResponse(results=recognition_results, media_url=media_url)


@router.post("/recognize/video", response_model=RecognitionMediaResponse)
async def recognize_faces_in_video(
    request: Request,
    user=Depends(get_authenticated_user),
    face_form: FaceForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    video_bytes = await face_form.file.read()

    if not video_bytes:
        raise HTTPException(status_code=400, detail="Empty video file")

    original_filename = face_form.file.filename or "video.mp4"

    suffix = Path(original_filename).suffix.lower()

    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        suffix = ".mp4"

    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(video_bytes)
        temp_video_path = Path(temp_file.name)

    video_capture = None
    video_writer = None

    try:

        video_capture = cv2.VideoCapture(str(temp_video_path))

        if not video_capture.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Unable to open video file",
            )

        fps = video_capture.get(cv2.CAP_PROP_FPS)

        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            fps = 25.0

        if width <= 0 or height <= 0:
            raise HTTPException(
                status_code=400, detail="Unable to determine video dimensions"
            )

        output_filename = f"video_{uuid4().hex}.mp4"
        output_path = settings.MEDIA_DIR / "recognition" / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        video_writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height),
        )

        if not video_writer.isOpened():
            raise HTTPException(status_code=500, detail="Unable to create output video")

        frame_index = 0

        # Keeps best recognition for each person
        recognized_people = {}

        # Best unknown result if any
        best_unknown = None

        # Keep boxes between processed frames
        last_annotations = []

        while True:
            ret, frame = video_capture.read()

            if not ret:
                break

            if frame_index % settings.VIDEO_FRAME_INTERVAL == 0:
                current_annotations = []

                try:
                    detected_faces = await run_in_threadpool(
                        DeepFace.represent,
                        img_path=frame,
                        model_name=(settings.FACE_EMBEDDING_MODEL),
                        enforce_detection=True,
                        detector_backend="opencv",
                    )

                except Exception:
                    detected_faces = []

                for detected_face in detected_faces:
                    embedding = detected_face["embedding"]

                    facial_area = detected_face["facial_area"]

                    face, distance = await find_best_face_match(
                        embedding=embedding, db=db
                    )

                    is_match = (
                        face is not None
                        and distance is not None
                        and distance < settings.MATCH_THRESHOLD
                    )

                    confidence = (
                        0.0 if distance is None else max(0.0, min(1.0, 1 - distance))
                    )

                    if is_match:
                        label = f"{face.person.name} " f"{confidence:.2%}"

                        color = (0, 255, 0)

                        result = RecognitionResponse(
                            person_id=face.person.id,
                            name=face.person.name,
                            confidence=confidence,
                            message="Match found",
                        )

                        existing = recognized_people.get(face.person.id)

                        if existing is None or confidence > existing.confidence:
                            recognized_people[face.person.id] = result

                    else:
                        label = "Unknown"

                        color = (0, 0, 255)

                        unknown_result = RecognitionResponse(
                            person_id=None,
                            name="Unknown",
                            confidence=confidence,
                            message="No match found",
                        )

                        if best_unknown is None or confidence > best_unknown.confidence:
                            best_unknown = unknown_result

                    current_annotations.append(
                        {"area": facial_area, "label": label, "color": color}
                    )

                last_annotations = current_annotations

            for annotation in last_annotations:
                area = annotation["area"]
                label = annotation["label"]
                color = annotation["color"]

                x = area["x"]
                y = area["y"]
                w = area["w"]
                h = area["h"]

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                cv2.putText(
                    frame,
                    label,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            video_writer.write(frame)

            frame_index += 1

        recognition_results = list(recognized_people.values())

        if best_unknown is not None:
            recognition_results.append(best_unknown)

        media_url = str(request.url_for("media", path=f"recognition/{output_filename}"))

        # Save recognition results
        recognition_result = RecognitionResult(
            ran_by=user.id if user else None,
            media_path=str(output_path),
            recognition_results=[result.model_dump() for result in recognition_results],
        )
        db.add(recognition_result)
        await db.commit()

        return RecognitionMediaResponse(
            results=recognition_results, media_url=media_url
        )

    finally:
        if video_capture is not None:
            video_capture.release()

        if video_writer is not None:
            video_writer.release()

        temp_video_path.unlink(missing_ok=True)


@router.get("/history", response_model=list[RecognitionMediaResponse])
async def get_recognition_history(
    request: Request,
    user=Depends(get_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    result = await db.execute(
        select(RecognitionResult).where(RecognitionResult.ran_by == user.id)
    )

    recognition_results = result.scalars().all()

    response = []
    for record in recognition_results:
        recognition_results = record.recognition_results
        response.append(
            RecognitionMediaResponse(
                results=[RecognitionResponse(**res) for res in recognition_results],
                media_url=str(
                    request.url_for(
                        "media", path=f"recognition/{Path(record.media_path).name}"
                    )
                ),
            )
        )

    return response
