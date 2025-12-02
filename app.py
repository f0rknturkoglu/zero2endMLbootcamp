import sys
import os

# src klasörünü path'e ekle
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.app import create_gradio_interface, inference

if __name__ == "__main__":
    print("Model yukleniyor...")
    try:
        inference.load_model()
        print("Model basariyla yuklendi!")
    except Exception as e:
        print(f"Model yuklenirken hata: {e}")
        
    interface = create_gradio_interface()
    interface.launch()
