from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.responses import RedirectResponse
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import overpy
import os
import uvicorn
from WasteClassification.chat import config, inverse_kinematics, process_image, run_chat_model
#from Biomedical_Imaging.engine import run_model
import tensorflow as tf
from PipelineCrack.engine import get_crack_result
from PipelineCrack.signal1 import *

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#mri_model = tf.keras.models.load_model("Biomedical_Imaging/MRI/DenseNet121_MRI.h5")
#xray_model = tf.keras.models.load_model("Biomedical_Imaging/XRay/DenseNet121_XRay.h5")

config()

@app.post("/mri")
async def predict(image: UploadFile = File(...)):
    tumor_type = await run_model("MRI", image)
    response_data = {"TumorType": tumor_type}
    print(response_data)
    return JSONResponse(content=response_data)

@app.post("/xray")
async def predict(image: UploadFile = File(...)):
    tumor_type = await run_model("XRay", image)
    response_data = {"TumorType": tumor_type}
    print(response_data)
    return JSONResponse(content=response_data)

@app.get("/", response_class=RedirectResponse)
async def redirect():
    return RedirectResponse(url="http://localhost:3000/home")

api = overpy.Overpass()

@app.get("/search_poi")
async def search_poi(
    q: str = Query(..., description="Query term for the POI, e.g., 'hospital'"),
    lat: float = Query(..., description="Latitude of the search center"),
    lon: float = Query(..., description="Longitude of the search center"),
    radius: int = Query(1000, description="Radius in meters for the search"),
    limit: int = Query(10, description="Limit the number of results")
):
    overpass_query = f"""
    [out:json][timeout:25];
    nwr(around:{radius},{lat},{lon})["amenity"="{q}"];
    out center;
    """
    response = api.query(overpass_query)
    result = []
    for node in response.nodes:
        name = node.tags.get("name", "n/a")
        if(name != "n/a"):
            result.append(name)

    return result[:limit]

@app.post("/process_image")
async def process(image: UploadFile = File(...)):
    img_str, file = await process_image(image)

    classification_result = run_chat_model(img_str, file)

    return JSONResponse({
        'classification': classification_result,
    })

@app.get("/crack_result")
async def crack_result():
    #result = "Crack" if get_crack_result() == 1.0 else "Normal"
    return JSONResponse(
        
        {"id": 1, "has_crack": "Crack"}
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)