from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse

from fastai.vision import *

from pathlib import Path
from io import BytesIO
import uvicorn
import aiohttp
import asyncio
import pandas as pd

app = Starlette()

path = Path('./')
learner = load_learner(path)

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)


def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    _,_,losses = learner.predict(img)
    return JSONResponse({
        "predictions": sorted(
            zip(learner.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    })


@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=UA-152392799-2"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());

          gtag('config', 'UA-152392799-2');
        </script>

            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <!-- Latest compiled and minified CSS -->
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
            <title>Detect Eye Diseases</title>
        </head>
        <body>
            <div class="container-fluid">
                <div class="row">
                    <div class="col-md-2">
                    </div>
                    <div class="col-md-8">
                        <h1>Detect Eye Diseases with Deep Learning Technology</h1>
                        <h2>This example is based on the fast.ai deep learning framework: <a href="https://www.fast.ai/">https://www.fast.ai/</a></h2>
                        <p><strong>Image classifier that detects different categories of eye diseases:<strong>
                            <ul class="list-group">
                                <li class="list-group-item">Normal Eye</li>
                                <li class="list-group-item">Glaucoma</li>
                                <li class="list-group-item">Retina Disease</li>
                                <li class="list-group-item">Cataract</li>
                            </ul>
                        </p>
                        <form action="/upload" method="post" enctype="multipart/form-data">
                            <div class="form-group">
                                Select image to upload:
                                <input type="file" name="file" class="input-sm">
                                <input type="submit" value="Upload and Analyze Image" class="btn btn-primary">
                            </div>
                        </form>
                    </div>
                    <div class="col-md-2">
                    </div>
                </div>
                <div class="row">
                    <div class="col-md-2">
                    </div>
                    <div class="col-md-8">
                        Or submit a URL:
                        <form action="/classify-url" method="get">
                            <div class="form-group">
                                <input type="url" name="url" class="input-sm">
                                <input type="submit" value="Fetch and Analyze image" class="btn btn-primary">
                            </div>
                        </form>
                    </div>
                    <div class="col-md-2">
                    </div>
                </div>
            </div>
        </body>
        </html>
    """)


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8008)
