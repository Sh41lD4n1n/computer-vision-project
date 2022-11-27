import torch
from fastapi import FastAPI, UploadFile, File, logger as fa_logger
from PIL import Image
from typing import List
from pandas import DataFrame

app = FastAPI(title='Computer Vision')

model = torch.hub.load('ultralytics/yolov5', 'custom', autoshape=True, path='./best.pt')
model.eval()

logger = fa_logger.logger


def predict_bboxes(img: List[Image.Image]) -> List[DataFrame]:
    """
    predicting bboxes for all images using yolo model
    :param img: list[Image.Image]
    :return: list[pd.DatafFrame]
    """
    def make_bbox(df):
        """
        filter bbox by confidence
        :param df: pd.DataFrame
        :return: dataframe with bbox
        """
        filtered_df = df[df['confidence'] > 0.5]
        bbox = filtered_df[['xmin', 'ymin', 'xmax', 'ymax']]
        return bbox

    data = model(img)
    bboxes = [make_bbox(im) for im in data.pandas().xyxy]
    return bboxes


def count_animals(old_bboxes: DataFrame, new_bboxes: DataFrame) -> int:
    """
    Count animals by difference between 2 dataframe with bboxes.
    It returns number of new animals in new image
    :param old_bboxes: pd.DataFrame
    :param new_bboxes: pd.DataFrame
    :return: int
    """
    def cond(x, y) -> bool:
        """
        return condition whether 2 bboxes are similar
        Similarity can be changed, but for now it is manhattan distance between bboxes less than threshold.
        :param x: iloc indexer of pd.DataFrame, indicating bbox
        :param y: iloc indexer of pd.DataFrame, indicating bbox
        :return: bool
        """
        threshold = 1500
        x = torch.from_numpy(x.to_numpy())
        y = torch.from_numpy(y.to_numpy())
        s = torch.sum(torch.abs(x - y)).item()
        return s < threshold

    # temporary data structures that saves whether certain bbox in similar to bbox from previous image
    bbox = new_bboxes.copy()
    prev_bbox = old_bboxes.copy()
    prev_bbox['prev'] = False
    bbox['prev'] = False
    for i in range(len(old_bboxes)):
        for j in range(len(new_bboxes)):
            # if bbox is already similar to bbox from previous image, then we don't need to check this bbox
            if bbox.iloc[j]['prev'] or prev_bbox.iloc[i]['prev']:
                continue
                # similarity between bboxes from previous and present images
            if cond(old_bboxes.iloc[i], new_bboxes.iloc[j]):
                # save status of similarity
                prev_bbox['prev'][i] = True
                bbox['prev'][j] = True
                # return number of non-similar(new animals) bboxes
    return bbox[bbox['prev'] == False].size.item()


def predict_sequence(bboxes: List[DataFrame]):
    # first image's bboxes are all animals
    count = bboxes[0].shape[0]
    logger.info(f'image 0: {count} new animals')
    for i in range(len(bboxes) - 1):
        # check the difference between two images and return new animals
        count += count_animals(bboxes[i], bboxes[i + 1])
        logger.info(f'image {i + 1}: {count} new animals')
    return count


@app.post('/predict/image')
def predict(files: List[UploadFile] = File(...)):
    """
    Prediction of number of animals in sequence of images.
    :param files: list[str], str is path to file
    :return: int
    """
    if not all([file.filename.split('.')[-1] in ['jpg', 'jpeg', 'png'] for file in files]):
        return 'Image must be jpg or png format!'
    images = [Image.open(file.file) for file in files]
    sequence = predict_bboxes(images)
    return predict_sequence(sequence)
