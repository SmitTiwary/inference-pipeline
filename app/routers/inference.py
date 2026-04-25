import uuid
from typing import List

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.utils.cleaner import clean_text

router = APIRouter(tags=["inference"])


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=32_000)


class ClassifyResponse(BaseModel):
    request_id: str
    label: str
    confidence: float
    input_tokens: int


class BatchClassifyRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=32)


class BatchClassifyResponse(BaseModel):
    outputs: List[ClassifyResponse]


@router.post("/classify", response_model=ClassifyResponse)
async def classify(body: ClassifyRequest, request: Request):
    text = clean_text(body.text)
    model = request.app.state.model
    label, confidence, n_tokens = model.classify(text)
    request.app.state.record_prediction(label, confidence, n_tokens)
    return ClassifyResponse(
        request_id=_new_id(),
        label=label,
        confidence=confidence,
        input_tokens=n_tokens,
    )


@router.post("/classify/batch", response_model=BatchClassifyResponse)
async def classify_batch(body: BatchClassifyRequest, request: Request):
    model = request.app.state.model
    outputs = []
    for raw in body.texts:
        text = clean_text(raw)
        label, confidence, n_tokens = model.classify(text)
        request.app.state.record_prediction(label, confidence, n_tokens)
        outputs.append(
            ClassifyResponse(
                request_id=_new_id(),
                label=label,
                confidence=confidence,
                input_tokens=n_tokens,
            )
        )
    return BatchClassifyResponse(outputs=outputs)


def _new_id() -> str:
    return str(uuid.uuid4())
