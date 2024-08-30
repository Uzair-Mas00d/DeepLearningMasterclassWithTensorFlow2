from fastapi import FastAPI
from service.api.api import main_router
import onnxruntime as rt

app = FastAPI(project_nmae="Emotion Detection")

app.include_router(main_router)

providers = ["CPUExecutionProvider"]
m_q = rt.InferenceSession("service/vit_keras_quant.onnx", providers=providers)

@app.get("/")
async def root():
    return  {"hello": "World"}