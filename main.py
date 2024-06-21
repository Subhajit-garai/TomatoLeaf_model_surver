import uvicorn
from fastapi import FastAPI,Request,responses,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
import model
from dotenv import load_dotenv
import os

app =FastAPI()
load_dotenv()
# Configure CORS
cors_origins = os.getenv("Origins")
origins = [
    cors_origins,
    # add more origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def SurverStatus():
    
     return {"message": "Running"}


@app.post("/predict")
async def predction(file: UploadFile=File(...)):
    data = await file.read()
    predclass,confident= model.TomatoLeaf_D_pred(data)    
    return responses.JSONResponse(content={
        "class":predclass,
        "confident": float(confident),
    })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)