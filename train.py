from ultralytics import YOLO


def main():
    model = YOLO('yolov8n.pt')
    model.train(data='datasets/class.yaml', epochs=100)


if __name__ == '__main__':
    main()
