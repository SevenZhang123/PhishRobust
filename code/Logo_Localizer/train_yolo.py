# Training a YOLO model on a newly transformed dataset
from ultralytics import YOLO

def train():
    # 1. Loading Model
    # Use yolo11m (Medium)
    model = YOLO('yolo11m.pt')

    # 2. Start training
    model.train(
        data='logo.yaml',   # Configuration file path
        epochs=100,         # Number of training rounds (recommended 50-100)
        imgsz=1280,         # Critical! The webpage screenshot is very large; a high resolution is required.
        batch=4,
        workers=4,          # CPU data read thread count
        device='0',
        project='runs/train',
        name='yolo_logo_exp1',
        patience=20,
        exist_ok=True
    )

if __name__ == '__main__':
    train()