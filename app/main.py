from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from .automl import train_automl, predict_new_data

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Train the model and get the best result on startup
best_result = train_automl()

@app.get("/")
def read_root():
    return {"Visit to Visualized Outcomes": "localhost:80/best_result/",
            "Visit to New Inputs":"localhost:80/predict/"}

@app.get("/best_result/")
def get_best_result(request: Request):
    return templates.TemplateResponse("visualizations.html", {"request": request, "result": best_result})

@app.post("/predict/")
def predict(request: Request, sepal_length: float = Form(...), sepal_width: float = Form(...), petal_length: float = Form(...), petal_width: float = Form(...)):
    new_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    updated_result = predict_new_data(new_data)
    return templates.TemplateResponse("visualizations.html", {"request": request, "result": updated_result})
