from fastapi import FastAPI

app = FastAPI(title='Computer Vision')


@app.get('/predict')
def predict(img):
    return img
