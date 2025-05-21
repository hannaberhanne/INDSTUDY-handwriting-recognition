# Converts PDF files to PNG images with security overrides and optimized processing
# Saves output to /data/raw/[destination_folder]/

import os
from PIL import Image
from pdf2image import convert_from_path
import warnings

# Disable security limits and warnings
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

def safe_pdf_to_png(pdf_path, output_folder, dpi=72, max_size=(1654, 2340)):
    """
    Convert PDF to PNG images with safety measures
    :param pdf_path: Path to source PDF file
    :param output_folder: Destination folder for PNG images
    :param dpi: Resolution density (default 72)
    :param max_size: Maximum (width, height) in pixels
    """
    try:
        # Verify PDF exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        print(f"Processing: {os.path.basename(pdf_path)}")

        # Get total page count
        from pdf2image.pdf2image import pdfinfo_from_path
        info = pdfinfo_from_path(pdf_path, userpw=None, poppler_path=None)
        total_pages = info["Pages"]

        # Process pages individually
        for page in range(1, total_pages + 1):
            try:
                # Convert single page to JPEG first
                jpeg_images = convert_from_path(
                    pdf_path,
                    dpi=dpi,
                    size=max_size,
                    first_page=page,
                    last_page=page,
                    fmt='jpeg',
                    use_pdftocairo=True,
                    thread_count=1
                )

                # Convert JPEG to PNG
                if jpeg_images:
                    png_path = os.path.join(output_folder, f"page_{page:02d}.png")
                    jpeg_images[0].save(png_path, "PNG", optimize=True)
                    print(f"Created: {png_path}")

            except Exception as page_error:
                print(f"Error processing page {page}: {str(page_error)}")
                continue

    except Exception as main_error:
        print(f"Conversion failed: {str(main_error)}")

if __name__ == "__main__":
    # Configure paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    # Process files with conservative settings
    safe_pdf_to_png(
        pdf_path=os.path.join(DATA_DIR, "Example Page - NHOC data.pdf"),
        output_folder=os.path.join(DATA_DIR, "raw", "example_page"),
        dpi=72,
        max_size=(1654, 2340)  # A4 size at 72 DPI
    )

    safe_pdf_to_png(
        pdf_path=os.path.join(DATA_DIR, "NHOC Ahafo Region.pdf"),
        output_folder=os.path.join(DATA_DIR, "raw", "ahafo_region"),
        dpi=72,
        max_size=(1654, 2340)
    )

    print("Conversion process completed")