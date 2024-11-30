from rotation_classifier import ECGClassifier
from PIL import Image

classifier = ECGClassifier().cuda()

img = Image.open(r"ECGs retos\ECGs retos\15105309.jpeg")
img = img.rotate(90)

print("Rotation was:", classifier(img)[0])
