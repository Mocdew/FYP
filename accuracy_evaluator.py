import csv
import pytesseract
from PIL import Image
import numpy as np

class AccuracyEvaluator:
    def __init__(self):
        self.accuracy_results = []

    def evaluate_accuracy(self, extracted_data, processed_images, selected_rows, selected_columns):
        """Calculate accuracy using Tesseract confidence scores for extracted table data."""
        self.accuracy_results = []
        try:
            for idx, (image_name, table_data) in enumerate(extracted_data):
                if not table_data or not selected_rows[idx] or not selected_columns[idx]:
                    continue
                confidences = []
                processed_image = processed_images[idx]
                for row_idx, (y_start, y_end) in enumerate(selected_rows[idx]):
                    if row_idx >= len(table_data):
                        continue
                    for col_idx, (x_start, x_end) in enumerate(selected_columns[idx]):
                        if col_idx >= len(table_data[row_idx]):
                            continue
                        cell_img = processed_image[y_start:y_end, x_start:x_end]
                        if cell_img.size == 0:
                            continue
                        pil_cell_img = Image.fromarray(cell_img)
                        # Get OCR data with confidence scores
                        ocr_data = pytesseract.image_to_data(pil_cell_img, config='--psm 6', output_type=pytesseract.Output.DICT)
                        # Filter for valid confidence scores (ignore -1, which indicates no text)
                        valid_confidences = [conf for conf in ocr_data['conf'] if conf >= 0]
                        if valid_confidences:
                            avg_confidence = sum(valid_confidences) / len(valid_confidences)
                            confidences.append(avg_confidence / 100.0)  # Scale to 0–1
                avg_accuracy = sum(confidences) / len(confidences) if confidences else 0.0
                self.accuracy_results.append((image_name, avg_accuracy))
            return self.accuracy_results
        except Exception as e:
            print(f"Accuracy evaluation failed: {str(e)}")
            return []

    def save_accuracy_to_csv(self, output_path):
        """Save accuracy results to a CSV file."""
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Image Name', 'Average Confidence (Accuracy Estimate)'])
                for image_name, accuracy in self.accuracy_results:
                    writer.writerow([image_name, f"{accuracy:.4f}"])
            return True
        except Exception as e:
            print(f"Failed to save accuracy results: {str(e)}")
            return False