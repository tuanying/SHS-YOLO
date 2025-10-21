from ultralytics.models import YOLO


if __name__ == '__main__':
    model = YOLO(r'/mnt/workspace/SHS-YOLO/ultralytics/cfg/models/11/SHS-YOLO.yaml')  # 此处以 m 为例，只需写yolov11m即可定位到m模型
    model.train(data=r'/mnt/workspace/pv2/data.yaml',
                imgsz=640,
                epochs=200,
                single_cls=False,  # 多类别设置False
                batch=64,
                workers=8,
                device='0'
                )
                
