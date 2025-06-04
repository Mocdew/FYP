# performance_metrics.py

import time
import csv
from difflib import SequenceMatcher

def measure_accuracy(extracted_text, ground_truth_text):
    """
    Simple accuracy score based on sequence similarity.
    """
    matcher = SequenceMatcher(None, extracted_text, ground_truth_text)
    return matcher.ratio() * 100

def measure_processing_time(processing_function, *args, **kwargs):
    """
    Measures time taken by a processing function.
    """
    start_time = time.time()
    result = processing_function(*args, **kwargs)
    end_time = time.time()
    processing_time = end_time - start_time
    return result, processing_time

def save_metrics_to_csv(metrics, csv_path):
    """
    Saves a list of metrics dicts to a CSV file.
    """
    fieldnames = ['Image Name', 'Processing Time (s)', 'Accuracy (%)']
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric)
