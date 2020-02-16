import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
import nest_asyncio

nest_asyncio.apply()
export_file_url = 'https://drive.google.com/uc?export=download&id=1U6vmC0eY_ejOvFvHIjXUsvI7Jsn31SRd'
export_file_name = 'export.pkl'

classes = ['cataract', 'glaucoma', 'normal', 'retina_disease']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))
#app.mount('/', StaticFiles(directory='/'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

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
    _,_,losses = learn.predict(img)    
    return JSONResponse({
        "predictions": sorted(
            zip(learner.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    })


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())

@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
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


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
