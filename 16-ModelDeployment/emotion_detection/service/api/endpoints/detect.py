from fastapi import APIRouter, UploadFile, HTTPException
from PIL import Image
from io import BytesIO
import numpy as np
from service.core.logic.onnx_inference import emotion_detector
from service.core.schema.output import APIOutput

emo_router = APIRouter()

@emo_router.post("/detect", response_model=APIOutput)
async def detect(im: UploadFile):
    if im.filename.split(".")[-1] in ("png", "jpg", "jpeg"):
        pass
    else:
        raise HTTPException(status_code=415, detail="No an image")
    
    image = Image.open(BytesIO(im.file.read()))
    image = np.array(image)
    
    return emotion_detector(image)