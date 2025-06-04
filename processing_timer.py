import time
import csv
from pathlib import Path

class ProcessingTimer:
    def __init__(self):
        self.timing_results = []

    def start_timer(self):
        """Start a timer."""
        return time.time()

    def end_timer(self, start_time):
        """End a timer and return elapsed time."""
        return time.time() - start_time

    def log_operation(self, image_name, operation, duration):
        """Log the duration of an operation for an image."""
        self.timing_results.append((image_name, operation, duration))

    def save_timing_to_csv(self, output_path):
        """Save timing results to a CSV file."""
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Image Name', 'Operation', 'Time (seconds)'])
                for image_name, operation, duration in self.timing_results:
                    writer.writerow([image_name, operation, f"{duration:.4f}"])
            return True
        except Exception as e:
            print(f"Failed to save timing results: {str(e)}")
            return False