import sys
import os

# src klasörünü path'e ekle
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.app import create_gradio_interface

if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch()
