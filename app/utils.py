import cv2
import numpy as np
from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.config.database import get_db
from app.config.settings import settings
from app.models import Face


async def find_best_face_match(
    embedding: list[float],
    db: AsyncSession = Depends(get_db),
):
    distance_expression = Face.embedding.cosine_distance(embedding)

    stmt = (
        select(Face, distance_expression.label("distance"))
        .options(joinedload(Face.person))
        .order_by(distance_expression)
        .limit(1)
    )

    result = await db.execute(stmt)
    row = result.first()

    if row is None:
        return None, None

    face, distance = row

    return face, float(distance)


def draw_face_result(
    image: np.ndarray,
    facial_area: dict,
    face,
    distance: float | None,
):
    x = facial_area["x"]
    y = facial_area["y"]
    w = facial_area["w"]
    h = facial_area["h"]

    is_match = (
        face is not None
        and distance is not None
        and distance < settings.MATCH_THRESHOLD
    )

    if is_match:
        confidence = max(0.0, min(1.0, 1 - distance))
        label = f"{face.person.name} {confidence:.2%}"
        color = (0, 255, 0)
    else:
        confidence = 0.0 if distance is None else max(0.0, min(1.0, 1 - distance))
        label = "Unknown"
        color = (0, 0, 255)

    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    cv2.putText(
        image,
        label,
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )

    return is_match, confidence
