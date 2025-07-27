import cv2
import numpy as np
from PIL import Image
from img2table.ocr import TesseractOCR
from img2table.document import Image as Img2TableImage

class TableDetector:
    def __init__(self):
        self.ocr = TesseractOCR(n_threads=1, lang="eng")

    def detect_tables(self, image):
        """
        Detects tables in an image using img2table.
        Args:
            image: numpy array or PIL Image
        Returns:
            list: List of dictionaries containing table information
                  Each dict has 'bbox' (x,y,w,h) and 'cells' list
        """
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            # Create img2table Image object
            doc = Img2TableImage(image)
            # Extract tables
            tables = doc.extract_tables(
                ocr=self.ocr,
                implicit_rows=True,
                borderless_tables=True,
                min_confidence=50
            )
            if not tables:
                return []
            result = []
            for table in tables:
                # Get table bounding box
                x1 = min(cell.bbox.x1 for row in table.content.values() for cell in row)
                y1 = min(cell.bbox.y1 for row in table.content.values() for cell in row)
                x2 = max(cell.bbox.x2 for row in table.content.values() for cell in row)
                y2 = max(cell.bbox.y2 for row in table.content.values() for cell in row)
                # Convert cells to expected format
                cells = []
                for row in table.content.values():
                    for cell in row:
                        cells.append({
                            'bbox': (cell.bbox.x1, cell.bbox.y1, cell.bbox.x2 - cell.bbox.x1, cell.bbox.y2 - cell.bbox.y1),
                            'content': getattr(cell, 'value', '')
                        })
                result.append({
                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                    'cells': cells
                })
            return result
        except Exception as e:
            print(f"Table detection error: {str(e)}")
            return []

    def _save_debug_visualization(self, image, rows, cols, temp_path):
        """Save a visualization of detected table structure."""
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            # Create a copy for visualization
            vis_img = image.copy()
            # Draw row lines
            for y1, y2 in rows:
                cv2.line(vis_img, (0, int(y1)), (vis_img.shape[1], int(y1)), (0, 255, 0), 2)
                cv2.line(vis_img, (0, int(y2)), (vis_img.shape[1], int(y2)), (0, 255, 0), 2)
            # Draw column lines
            for x1, x2 in cols:
                cv2.line(vis_img, (int(x1), 0), (int(x1), vis_img.shape[0]), (0, 255, 0), 2)
                cv2.line(vis_img, (int(x2), 0), (int(x2), vis_img.shape[0]), (0, 255, 0), 2)
            # Save the visualization
            cv2.imwrite(temp_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"Debug visualization error: {str(e)}") 