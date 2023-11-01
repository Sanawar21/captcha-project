# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import torch


class Predictor(BasePredictor):

    CLASSES = ['M', 'Y', '8', '9', 'F', 'B', 'V', 'I', 'Q', 'H', '4', 'P', 'T',
               'C', 'W', 'A', 'K', 'G', 'N', 'L', '5', '6', '2', '0', 'Z', '7', '1', 'J', 'D', 'E',
               'O', 'X', '3', 'R']

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = torch.hub.load(
            "WongKinYiu/yolov7", "custom", "captcha_model.pt", trust_repo=True)
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        image: Path = Input(description="Input image"),
    ) -> Path:
        """Run a single prediction on the model"""
        results = self.model(image, size=640)
        predictions = results.pred[0]
        boxes = list(predictions[:, :4])
        categories = [int(x) for x in list(predictions[:, 5])]

        string = ''
        cat_and_pos = []

        for i in range(len(categories)):
            box = boxes[i]
            cat = self.CLASSES[categories[i]]
            cat_and_pos.append((cat, float(box[0])))

        cat_and_pos.sort(key=lambda x: x[1])

        for cat, _ in cat_and_pos:
            string += cat

        return string.replace("O", "0")
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
