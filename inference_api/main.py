from fastapi import FastAPI, HTTPException, File, UploadFile
from inference_api.model_service import ModelService

app = FastAPI()

@app.get("/")
def read_root():
  # raise HTTPException(status_code=400, detail=f'Hello! You can use /inference/text to run infrence engine')
  return {"hello": "Use /inference/text to run the model with the given text"}

@app.get("/inference/{text_to_classify}")
def inference(text_to_classify: str):
  response = ModelService.predict(text_to_classify)
  # return response
  return response['classes']
