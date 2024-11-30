import torch
from torch import nn
from torchvision.transforms.v2 import functional as F
from torchvision.models import mobilenet_v3_small
import numpy as np
from typing import Union, List, Tuple
from PIL import Image

ImageType = Union[Image.Image, np.ndarray]


class ECGRotationClassifier:
    """Classifies whether an ECG was rotated and by how much in increments of +-90 degrees counterclockwise.
    Classification options: [-90, 0, 90, 180].
    """

    def __init__(self, model_path: str) -> None:
        self.model = mobilenet_v3_small()
        self.model.classifier += nn.Sequential(
            nn.Linear(self.model.classifier.pop(-1).in_features, 4), nn.Softmax(-1)
        )
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

        self.rotation_options = [-90, 0, 90, 180]
        self.use_cuda = False

    def _img_to_tensor(self, img: Union[np.ndarray, Image.Image]):
        if isinstance(img, Image.Image):
            return F.pil_to_tensor(img)
        else:
            return torch.from_numpy(img).permute((2, 0, 1))

    def _normalize_img(self, img: torch.Tensor) -> torch.Tensor:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        data = img.to(dtype=torch.float32) / 255.0

        return F.normalize(data, mean, std)

    def _resize_tensor(
        self, img: torch.Tensor, size_of_smallest_side: int
    ) -> torch.Tensor:
        height, width = img.shape[1:]
        r = min(height, width) / size_of_smallest_side

        new_height, new_width = 0, 0
        if height < width:
            new_height = int(size_of_smallest_side)
            new_width = int(width / r)
        else:
            new_width = int(size_of_smallest_side)
            new_height = int(height / r)

        return F.resize(img, (new_height, new_width))

    def _resize_tensor_512(self, img: torch.Tensor):
        return self._resize_tensor(img, 512)

    def _map_logits_to_rotation(self, logits: torch.Tensor):
        return self.rotation_options[logits.squeeze().argmax().item()]

    def cuda(self, id=0):
        self.model = self.model.cuda(id)
        self.use_cuda = True

        return self

    def cpu(self):
        self.model = self.model.cpu()
        self.use_cuda = False

        return self

    @torch.no_grad
    def __call__(self, img: Union[ImageType, List[ImageType]]) -> Tuple[int]:
        """Runs the model for the specified image or images.

        Args:
            img (Union[ImageType, List[ImageType]]): A list of images or a single image
                to be processed.

        Raises:
            TypeError: If the objected provided was not a List of images
                or a single image.

        Returns:
            Tuple[int]: The rotation for each image provided. It will always return a
                list even if the input was a single image and not a list.
        """

        batched_images = None

        if isinstance(img, list):
            torched_images = map(self._img_to_tensor, img)

            resized_images = map(self._resize_tensor_512, torched_images)

            norm_images = map(self._normalize_img, resized_images)

            batched_images = torch.stack(list(norm_images))
        elif isinstance(img, np.ndarray) or isinstance(img, Image.Image):
            torched_image = self._img_to_tensor(img)

            resized_image = self._resize_tensor_512(torched_image)

            norm_image = self._normalize_img(resized_image)

            batched_images = norm_image.unsqueeze(0)
        else:
            raise TypeError(
                f"Invalid type of {str(type(img))} was provided! Must be one of List[Image | ndarray], Image or ndarray"
            )

        if self.use_cuda:
            batched_images = batched_images.cuda()

        classification: torch.Tensor = self.model(batched_images)

        outputs = classification.chunk(classification.shape[0])

        return tuple(map(self._map_logits_to_rotation, outputs))


if __name__ == "__main__":
    classifier = ECGRotationClassifier(
        r"rotation_classifier\mobilenet_small-0-21985.pt"
    )
    img = Image.open(r"image.png")
    img = img.rotate(180, expand=1)

    print("Detected rotation:", classifier(img))
