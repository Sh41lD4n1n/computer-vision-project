import torch
from fastapi import FastAPI, UploadFile, File, logger as fa_logger
from PIL import Image
import uvicorn
from typing import List
from pandas import DataFrame

app = FastAPI(title='Computer Vision')

model = torch.hub.load('ultralytics/yolov5', 'custom', autoshape=True, path='./best.pt')
model.eval()

logger = fa_logger.logger


def predict_bboxes(img: List[Image.Image]) -> List[DataFrame]:
    def make_bbox(df):
        filtered_df = df[df['confidence'] > 0.3]
        bbox = filtered_df[['xmin', 'ymin', 'xmax', 'ymax']]
        return bbox

    data = model(img)
    bboxes = [make_bbox(im) for im in data.pandas().xyxy]
    return bboxes


def count_animals(old_bboxes: DataFrame, new_bboxes: DataFrame) -> int:
    def cond(x, y) -> bool:
        threshold = 1000
        x = torch.from_numpy(x.to_numpy())
        y = torch.from_numpy(y.to_numpy())
        s = torch.sum(torch.abs(x - y)).item()
        logger.info(s)
        return s < threshold

    bbox = new_bboxes.copy()
    bbox['prev'] = False
    for i in range(len(old_bboxes)):
        for j in range(len(new_bboxes)):
            if bbox.iloc[j]['prev']:
                continue
            if cond(old_bboxes.iloc[i], new_bboxes.iloc[j]):
                bbox['prev'][j] = True
    return bbox[bbox['prev'] == True].size.item()


def predict_sequence(bboxes: List[DataFrame]):
    count = bboxes[0].shape[0]
    logger.info(f'image 0: {count} new animals')
    for i in range(len(bboxes) - 1):
        count += count_animals(bboxes[i], bboxes[i + 1])
        logger.info(f'image {i + 1}: {count} new animals')
    return count


@app.post('/predict/image')
def predict(files: List[UploadFile] = File(...)):
    if not all([file.filename.split('.')[-1] in ['jpg', 'jpeg', 'png'] for file in files]):
        return 'Image must be jpg or png format!'
    images = [Image.open(file.file) for file in files]
    sequence = predict_bboxes(images)
    return predict_sequence(sequence)
