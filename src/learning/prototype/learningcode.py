from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")
    model.train(
        data="final_data4/data.yaml",
        epochs=500,
        imgsz=320,
        batch=32,
        workers=0, 

        # patience=20,  #과적합방지 (이후 epochs가 20번 돌렸음에도 성능향상x일시 조기종료 )
        optimizer="AdamW",
        amp=True,
        cos_lr=True,
        close_mosaic=10,

        fliplr=0.5,
        degrees=5.0,
        scale=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    metrics = model.val()

    print("\n=== Validation Metrics ===")
    print(f"Precision   : {metrics.results_dict.get('metrics/precision(B)', 'N/A')}")
    print(f"Recall      : {metrics.results_dict.get('metrics/recall(B)', 'N/A')}")
    print(f"mAP50       : {metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"mAP50-95    : {metrics.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

# 무한루프 방지용 넣어야함
if __name__ == "__main__":
    main()
