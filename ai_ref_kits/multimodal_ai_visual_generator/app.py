import io
import sys
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from PIL import Image
from PySide6.QtCore import Qt, QThread
from PySide6.QtCore import Signal
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QProgressBar, QTextEdit, QLineEdit
)
from PySide6.QtWidgets import QLabel
from torchvision.transforms import Compose

from vad_whisper_workers import VADWorker, WhisperWorker
from stable_diffusion_worker import StableDiffusionWorker
from multiprocessing import Queue


from queue import Empty
class ImageUpdateThread(QThread):
    progress_updated = Signal(int, str)
    primary_pixmap_updated = Signal(QPixmap)
    
    def __init__(self, sd_prompt_queue, app_params):
        super().__init__()
        
        self.running = True
        self.sd_prompt_queue = sd_prompt_queue
        self.sd_device = app_params["sd_device"]
        self.super_res_device = app_params["super_res_device"]
        
    def stop(self):
        self.running = False
        self.quit()
        self.wait()
        
    def run(self):
        sd_worker = StableDiffusionWorker(self.sd_prompt_queue, self.sd_device, self.super_res_device)
        produced_img_queue = sd_worker.result_img_queue
        sd_worker.start()
        
        while self.running:
            try:
                #wait for a new image from the sd worker
                img = produced_img_queue.get(timeout=1)
                
                # Convert the image buffer to QPixmap
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                buffer.seek(0)
                pixmap = QPixmap()
                pixmap.loadFromData(buffer.read(), "PNG")
                
                # finally, this updates the UI image. 
                self.primary_pixmap_updated.emit(pixmap)
 
            except Empty:
                continue  # Queue is empty, just wait
                
        sd_worker.stop()

from llm_worker import LLMWorker
class LLMWorkerThread(QThread):
    caption_updated = Signal(str)
    progress_updated = Signal(int, str)
    
    def __init__(self, transcription_queue, sd_prompt_queue, app_params, theme):
        super().__init__()
        
        self.running = True
        self.transcription_queue = transcription_queue
        self.sd_prompt_queue = sd_prompt_queue
        self.llm_device = app_params["llm_device"]
        self.theme = theme
        
    def stop(self):
        self.running = False
        self.quit()
        self.wait()
        
    def run(self):
        from multiprocessing import Queue
        ui_update_queue = Queue()
        
        llm_worker = LLMWorker(self.transcription_queue, self.sd_prompt_queue, ui_update_queue, self.llm_device, self.theme)
        llm_worker.start()
        
        while self.running:
            try:
                #wait for some UI update from the llm worker
                ui_update = ui_update_queue.get(timeout=1) 
                key = ui_update[0]
                value = ui_update[1]
                
                if key == "caption":
                    self.caption_updated.emit(value)
                elif key == "progress":
                    self.progress_updated.emit(0, value)
                    
            except Empty:
                continue  # Queue is empty, just wait
                
        
        llm_worker.stop()
        
class ClickableLabel(QLabel):
    clicked = Signal()  # Define a signal to emit on click

    def mousePressEvent(self, event):
        self.clicked.emit()  # Emit the clicked signal
        super().mousePressEvent(event)

class MainWindow(QMainWindow):
    def __init__(self, app_params):
        super().__init__()

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QGridLayout(self.central_widget)

        self.app_params = app_params

        # Image pane
        self.image_label = ClickableLabel("No Image")
        #self.image_label.setFixedSize(1280, 720)
        self.image_label.setFixedSize(1216, 684)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label, 0, 1)

        # Connect the click signal
        self.display_primary_img = True
        #self.image_label.clicked.connect(self.swap_image)

        self.primary_pixmap = None
        self.depth_pixmap = None

        # Caption
        self.caption_label = QLabel("No Caption")
        fantasy_font = QFont("Papyrus", 18, QFont.Bold)
        self.caption_label.setFont(fantasy_font)
        self.caption_label.setAlignment(Qt.AlignCenter)
        self.caption_label.setWordWrap(True)  # Enable word wrapping
        layout.addWidget(self.caption_label, 1, 1)

        # Log widget
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setStyleSheet("background-color: #f0f0f0; border: 1px solid gray;")
        layout.addWidget(self.log_widget, 0, 2, 2, 1)
        self.log_widget.hide()  # Initially hidden

        bottom_layout = QVBoxLayout()

        # Bottom pane with buttons and progress bar
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_thread)
        button_layout.addWidget(self.start_button)

        self.toggle_theme_button = QPushButton("Theme")
        self.toggle_theme_button.clicked.connect(self.toggle_theme)
        button_layout.addWidget(self.toggle_theme_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("Idle")
        self.progress_bar.setValue(0)
        button_layout.addWidget(self.progress_bar)

        bottom_layout.addLayout(button_layout)

        # Theme text box, initially hidden
        self.theme_input = QLineEdit()
        self.theme_input.setPlaceholderText("Enter a theme here...")
        self.theme_input.setText("Medieval Fantasty Adventure")
        self.theme_input.setStyleSheet("background-color: white; color: black;")
        self.theme_input.hide()
        bottom_layout.addWidget(self.theme_input)

        layout.addLayout(bottom_layout, 2, 0, 1, 3)

        # Worker threads
        self.speech_thread = None
        self.worker = None

        # Window configuration
        self.setWindowTitle("AI Adventure Experience")
        self.resize(800, 600)
        
        self.vad_worker = VADWorker()
        self.whisper_worker = WhisperWorker(self.vad_worker.result_queue, self.app_params["whisper_device"])
        self.transcription_queue = self.whisper_worker.result_queue
        self.sd_prompt_queue = Queue()
        
        #TODO: We need to be able to update the theme for the llm_worker.
        self.llm_worker = LLMWorkerThread(self.transcription_queue, self.sd_prompt_queue, self.app_params, self.theme_input.text())
        self.llm_worker.caption_updated.connect(self.update_caption)
        self.llm_worker.progress_updated.connect(self.update_progress)
        
        self.img_update_worker = ImageUpdateThread(self.sd_prompt_queue, self.app_params)
        self.img_update_worker.primary_pixmap_updated.connect(self.update_primary_pixmap)
        
        self.llm_worker.start()
        self.img_update_worker.start()
        
        self.listening = False

    def start_thread(self):
        if self.listening == False:
            self.vad_worker.start()
            self.whisper_worker.start()
            self.listening = True
            self.start_button.setText("Stop")

        else:
            self.vad_worker.stop()
            self.whisper_worker.stop()
            self.start_button.setText("Start")

    def toggle_log(self):
        if self.log_widget.isVisible():
            self.log_widget.hide()
        else:
            self.log_widget.show()

    def toggle_theme(self):
        if self.theme_input.isVisible():
            self.theme_input.hide()
        else:
            self.theme_input.show()

    def update_primary_pixmap(self, pixmap):
        self.primary_pixmap = pixmap
        self.update_image_label()

    def update_image_label(self):
        if self.primary_pixmap is not None:
            pixmap = self.primary_pixmap
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size()))

    def update_caption(self, caption):
        self.caption_label.setText(caption)
        #self.log_widget.append(f"Caption updated: {caption}")

    def update_progress(self, value, label):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(label)

    def closeEvent(self, event):
        if self.llm_worker and self.llm_worker.isRunning():
            self.llm_worker.stop()  # Gracefully stop the worker thread
            self.llm_worker.wait()  # Wait for the thread to finish
            
        if self.vad_worker:
            self.vad_worker.stop()
        
        if self.whisper_worker:
            self.whisper_worker.stop()
        
        if self.img_update_worker and self.img_update_worker.isRunning():
            self.img_update_worker.stop()

        event.accept()  # Proceed with closing the application


def main():
    app = QApplication(sys.argv)
    core = ov.Core()

    llm_device = "GPU" if "GPU" in core.available_devices else "CPU"
    sd_device = "GPU" if "GPU" in core.available_devices else "CPU"
    whisper_device = 'CPU'
    super_res_device = "GPU" if "GPU" in core.available_devices else "CPU"
    depth_anything_device = "GPU" if "GPU" in core.available_devices else "CPU"

    print("Just a minute... doing some application setup...")

    # create the 'results' folder if it doesn't exist
    Path("results").mkdir(exist_ok=True)

    app_params = {}
    app_params["llm_device"] = llm_device
    app_params["sd_device"] = sd_device
    app_params["super_res_device"] = super_res_device
    app_params["whisper_device"] = whisper_device

    print("Demo is ready!")
    window = MainWindow(app_params)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()