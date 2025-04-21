import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

import tensorflow as tf
from data_loader import load_metadata, load_images
from model import build_cnn

def train_and_evaluate(images, labels, label_map, num_layers, epochs=5, batch_size=32, results_file="results.txt"):
    # Stratifierad split för att bevara klassfördelning: 60% training, 20% validation, 20% test
    x_temp, x_test, y_temp, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels.argmax(axis=1))
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, stratify=y_temp.argmax(axis=1))

    model = build_cnn(num_layers, input_shape=x_train.shape[1:], num_classes=len(label_map))

    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(y_train.argmax(axis=1)),
                                         y=y_train.argmax(axis=1))
    class_weights_dict = dict(enumerate(class_weights))

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stop],
                        class_weight=class_weights_dict)

    loss, acc = model.evaluate(x_test, y_test)
    print(f"\nModel with {num_layers} layers - Test Accuracy: {acc:.4f}")

    # Skriver testresultat
    with open(results_file, 'a') as f:
        f.write(f"Model with {num_layers} layers - Test Accuracy: {acc:.4f}\n")

    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    report = classification_report(y_true, y_pred_classes, labels=np.unique(y_true),
                                   target_names=[k for k, v in label_map.items() if v in np.unique(y_true)])
    f1 = f1_score(y_true, y_pred_classes, average='weighted')
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    recall = recall_score(y_true, y_pred_classes, average='weighted')

    cm = confusion_matrix(y_true, y_pred_classes)
    print("Confusion Matrix:\n", cm)

    # Visualiserar confusion matrix med Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[k for k,v in label_map.items() if v in np.unique(y_true)],
                yticklabels=[k for k,v in label_map.items() if v in np.unique(y_true)])
    plt.title(f"Confusion Matrix - {num_layers} Layers")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


    print(report)
    print(f"F1-score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    with open(results_file, 'a') as f:
        f.write(f"Classification Report for {num_layers} layers:\n{report}\n")
        f.write(f"F1-score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}\n")
        f.write(f"Confusion Matrix for {num_layers} layers:\n{cm}\n")

    # Plot träningskurvor
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f"Accuracy - {num_layers} Layers")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    metadata_path = "./HAM10000_metadata.csv"
    image_dir = "./HAM10000_images"
    binary_mode = False 

    print("Laddar metadata...")
    df = load_metadata(metadata_path)

    print("Laddar bilder...")
    images, labels, label_map = load_images(df, image_dir, binary_classification=binary_mode)

    with open("results.txt", 'w') as f:
        f.write("Model Evaluation Results\n")
        f.write("="*30 + "\n")

    for num_layers in [1, 2, 3, 4, 5, 6]:
        print(f"\n--- Tränar modell med {num_layers} lager ---")
        train_and_evaluate(images, labels, label_map, num_layers=num_layers)

    print("\nAlla resultat har skrivits till results.txt")
