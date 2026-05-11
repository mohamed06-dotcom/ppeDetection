import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO


RESULTS_PATH = "runs/detect/train/results.csv"
MODEL_PATH = r"C:\Users\moham\PycharmProjects\ppeDetection\src\model\best-70ep.pt"

OUTPUT_DIR = "evaluation_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():

    df = pd.read_csv(RESULTS_PATH)

    df.columns = df.columns.str.strip()

    last = df.iloc[-1]

    print("\n===================================")
    print("        FINAL METRICS")
    print("===================================\n")

    print(f"Precision:   {last['metrics/precision(B)']:.4f}")
    print(f"Recall:      {last['metrics/recall(B)']:.4f}")
    print(f"mAP@50:      {last['metrics/mAP50(B)']:.4f}")
    print(f"mAP@50-95:   {last['metrics/mAP50-95(B)']:.4f}")

    # =========================================================
    # Epochs
    # =========================================================
    epochs = df["epoch"]

    # =========================================================
    # Training Loss Curves
    # =========================================================
    plt.figure(figsize=(10, 5))

    plt.plot(
        epochs,
        df["train/box_loss"],
        label="Box Loss"
    )

    plt.plot(
        epochs,
        df["train/cls_loss"],
        label="Classification Loss"
    )

    plt.plot(
        epochs,
        df["train/dfl_loss"],
        label="DFL Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)

    loss_curve_path = os.path.join(
        OUTPUT_DIR,
        "loss_curves.png"
    )

    plt.savefig(loss_curve_path, dpi=300)
    plt.close()

    print(f"\nSaved: {loss_curve_path}")

    plt.figure(figsize=(10, 5))

    plt.plot(
        epochs,
        df["metrics/mAP50(B)"],
        label="mAP@50"
    )

    plt.plot(
        epochs,
        df["metrics/mAP50-95(B)"],
        label="mAP@50-95"
    )

    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("mAP Curves")
    plt.legend()
    plt.grid(True)

    map_curve_path = os.path.join(
        OUTPUT_DIR,
        "map_curves.png"
    )

    plt.savefig(map_curve_path, dpi=300)
    plt.close()

    print(f"Saved: {map_curve_path}")
    print("\n===================================")
    print("     RUNNING YOLO VALIDATION")
    print("===================================\n")

    model = YOLO(MODEL_PATH)

    metrics = model.val(
        save=True,
        save_json=True,
        plots=True
    )

    print("\n===================================")
    print("       PER-CLASS mAP")
    print("===================================\n")

    rows = []

    for i, class_name in metrics.names.items():

        class_map50_95 = metrics.box.maps[i]

        rows.append({
            "Class": class_name,
            "mAP50-95": round(class_map50_95, 4)
        })

        print(
            f"{class_name:<15} "
            f"mAP50-95: {class_map50_95:.4f}"
        )

    # Save CSV
    metrics_df = pd.DataFrame(rows)

    class_metrics_path = os.path.join(
        OUTPUT_DIR,
        "class_metrics.csv"
    )

    metrics_df.to_csv(
        class_metrics_path,
        index=False
    )

    print(f"\nSaved: {class_metrics_path}")

    print("\n===================================")
    print("       EVALUATION COMPLETE")
    print("===================================\n")

    print("Generated files:")

    print(f"- {loss_curve_path}")
    print(f"- {map_curve_path}")
    print(f"- {class_metrics_path}")

    print("\nYOLO validation outputs saved inside:")
    print("runs/detect/val/")


if __name__ == "__main__":
    main()