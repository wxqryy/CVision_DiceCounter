from ultralytics import YOLO
from PIL import Image, ImageDraw

def diceAI(name):
    model = YOLO("best.pt")
    results = model.predict(name)
    result = results[0]
    img = Image.open(name)
    pencil = ImageDraw.Draw(img)
    counter = []
    values = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6}
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        pencil.rectangle((cords), outline='cyan')
        pencil.text(text=class_id, xy=(cords), fill='cyan', font_size=15)
        counter.append(values[class_id])
    return (counter)