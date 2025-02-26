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

LLM_SYSTEM_MESSAGE_START="""
You are a specialized helper bot designed to process live transcripts from a demo called "AI Adventure Game", which showcases a tabletop adventure game with live illustrations generated by a text-to-image model.

Your role is to act as a filter:

Detect descriptions of game scenes from the transcript that require illustration.
Output a detailed SD Prompt for these scenes.
When you detect a scene for the game, output it as:

SD Prompt: <a detailed prompt for illustration>

Guidelines:
Focus only on game scenes: Ignore meta-comments, explanations about the demo, or incomplete thoughts.
Contextual Awareness: Maintain and apply story context, such as the location, atmosphere, and objects, when crafting prompts. Update this context only when a new scene is explicitly described.
No Players in Prompts: Do not include references to "the player," "the players,"  "the party", or any specific characters in the SD Prompt. Focus solely on the environment and atmosphere.
Prioritize Clarity: If unsure whether the presenter is describing a scene, return: 'None'. Avoid making assumptions about incomplete descriptions.
Enhance Visuals: Add vivid and descriptive details to SD Prompts, such as lighting, mood, style, or texture, when appropriate, but stay faithful to the transcript.
Examples:
Example 1:
Input: "Let me explain how we are using AI for these illustrations." Output: 'None'

Example 2:
Input: "The party is standing at the gates of a large castle." Output: SD Prompt: "A massive medieval castle gate with towering stone walls, surrounded by mist and faintly glowing lanterns at dusk."

Example 3:
Context: "The party is at the gates of a large castle." Input: "The party then encounters a huge dragon." Output: SD Prompt: "A massive dragon with gleaming scales, standing before the misty gates of a towering medieval castle, lit by glowing lanterns under a dim sky."

Example 4:
Input: "And now the players roll for initiative." Output: 'None'

The presenter of the demo is aware of your presence and role, and will sometimes refer to you as the 'LLM', the 'agent', etc. Occasionally he will point out your roles and read back the SD prompts that you generate. When you detect this, return 'None'.

The SD prompts should be no longer than 25 words.

Only output SD prompts it is detected that there is big difference in location as compared with the last SD prompt that you gave.

Example 1:

Input 0: "The party is standing at the gates of a large castle." Output 0: SD Prompt: "A massive medieval castle gate with towering stone walls, surrounded by mist and faintly glowing lanterns at dusk."
Input 1: "A character is still at the gates of the castle." Output 1: 'None'

"""

LLM_SYSTEM_MESSAGE_END="""

Additional hints and reminders:
* You are a filter, not a chatbot. Only provide SD Prompts or 'None.'
* No Extra Notes: Do not include explanations, comments, or any text beyond the required SD Prompt or 'None.'
* Validate Completeness: A description of a scene often involves locations, objects, or atmosphere and is unlikely to be inferred from just verbs or generic phrases.
* If it seems that the transcription of the presenter is simply reading a previous SD prompt that you generated, return 'None'
* The SD prompts should be no longer than 25 words.
* Do not provide SD prompts for what seem like incomplete thoughts. Return 'None' in this case.
* Use the given theme of the game to help you decide whether or not the given bits of transcript are describing a new scene, or not.
* Do not try to actually illustrate the characters themselves, only details of their environmental surroundings & atmosphere.
* The SD prompts should be no longer than 25 words.
* Only output SD prompts it is detected that there is big difference in location as compared with the last SD prompt that you gave. If it seems like the location is the same, just return 'None'
"""

from queue import Empty
class ImageUpdateThread(QThread):
    image_updated = Signal(QPixmap)
    progress_updated = Signal(int, str)
    
    primary_pixmap_updated = Signal(QPixmap)
    depth_pixmap_updated = Signal(QPixmap)
    
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
        
class WorkerThread(QThread):
    image_updated = Signal(QPixmap)
    caption_updated = Signal(str)
    progress_updated = Signal(int, str)

    primary_pixmap_updated = Signal(QPixmap)
    depth_pixmap_updated = Signal(QPixmap)

    def __init__(self, queue, sd_prompt_queue, app_params, theme):
        super().__init__()

        self.running = True

        self.queue = queue
        self.llm_pipeline = app_params["llm"]
        self.theme = theme
        self.sd_prompt_queue = sd_prompt_queue

        print("theme: ", self.theme)

    def sd_callback(self, i, num_inference_steps, callback_userdata):
        if num_inference_steps > 0:
            prog = int((i / num_inference_steps) * 100)
            self.progress_updated.emit(prog, "illustrating")

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

    def llm_streamer(self, subword):
        #print(subword, end='', flush=True)
        self.stream_message += subword

        search_string = "SD Prompt:"
        if search_string in self.stream_message and 'None' not in self.stream_message:
            if self.stream_sd_prompt_index is None:
                self.stream_sd_prompt_index = self.stream_message.find(search_string)

            start_index = self.stream_sd_prompt_index
            # Calculate the start index of the new string (1 character past the ':')
            prompt = self.stream_message[start_index + len(search_string):].strip()

            self.caption_updated.emit(prompt)
        elif 'None' in self.stream_message:
            #Sometimes the LLM gives a response like: None (And then some long description why in parenthesis)
            # Basically, as soon as we see 'None', just stop generating tokens.
            return True

        # Return flag corresponds whether generation should be stopped.
        # False means continue generation.
        return False

    def run(self):

        llm_tokenizer = self.llm_pipeline.get_tokenizer()

        # Assemble the system message.
        system_message=LLM_SYSTEM_MESSAGE_START
        system_message+="\nThe presenter is giving a hint that the theme of their game is: " + self.theme
        system_message+="\nYou should use this hint to guide your decision about whether the presenter is describing a scene from the game, or not, and also to generate adequate SD Prompts."
        system_message+="\n" + LLM_SYSTEM_MESSAGE_END
        #print("System Message:")
        #print(system_message)


        generate_config = ov_genai.GenerationConfig()

        generate_config.temperature = 0.7
        generate_config.top_p = 0.95
        generate_config.max_length = 2048

        meaningful_message_pairs = []

        while self.running:
            try:
                # Wait for a sentence from the queue
                self.progress_updated.emit(0, "listening")

                result = self.queue.get(timeout=1)

                self.progress_updated.emit(0, "processing")

                chat_history = [{"role": "system", "content": system_message}]

                #only keep the latest 5 meaningful message pairs (last 2 illustrations)
                meaningful_message_pairs = meaningful_message_pairs[-2:]

                formatted_prompt = system_message
                
                #print("number of meaningful messages in history: ", len(meaningful_message_pairs))
                for meaningful_pair in meaningful_message_pairs:
                    user_message = meaningful_pair[0]
                    assistant_response = meaningful_pair[1]

                    chat_history.append({"role": "user", "content": user_message["content"]})
                    chat_history.append({"role": "assistant", "content": assistant_response["content"]})

                chat_history.append({"role": "user", "content": result})
                formatted_prompt = llm_tokenizer.apply_chat_template(history=chat_history, add_generation_prompt=True)

                self.progress_updated.emit(0, "processing...")
                self.stream_message=""
                self.stream_sd_prompt_index=None
                print("running llm!")
                llm_result = self.llm_pipeline.generate(inputs=formatted_prompt, generation_config=generate_config, streamer=self.llm_streamer)

                search_string = "SD Prompt:"

                #sometimes the llm will return 'SD Prompt: None', so filter out that case.
                if search_string in llm_result and 'None' not in llm_result:
                    # Find the start of the search string
                    start_index = llm_result.find(search_string)
                    # Calculate the start index of the new string (1 character past the ':')
                    prompt = llm_result[start_index + len(search_string):].strip()
                    #print(f"Extracted prompt: '{prompt}'")

                    caption = prompt
                    self.caption_updated.emit(caption)
                    #self.progress_updated.emit(0, "illustrating...")
                    self.sd_prompt_queue.put(prompt)

                    #self.generate_image(prompt)
                    #self.image_updated.emit(pixmap)  # Emit the QPixmap

                    # this was a meaningful message!
                    meaningful_message_pairs.append(
                    ({"role": "user", "content": result},
                     {"role": "assistant", "content": llm_result},)
                    )

            except Empty:
                continue  # Queue is empty, just wait

        self.progress_updated.emit(0, "idle")


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

        self.llm_pipeline = app_params["llm"]
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
        self.image_label.clicked.connect(self.swap_image)

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

    def start_thread(self):
        if not self.worker or not self.worker.isRunning():

            self.vad_worker = VADWorker()
            self.vad_worker.start()

            self.whisper_worker = WhisperWorker(self.vad_worker.result_queue, self.app_params["whisper_device"])
            self.whisper_worker.start()

            self.transcription_queue = self.whisper_worker.result_queue
            
            from multiprocessing import Queue
            self.sd_prompt_queue = Queue()

            self.worker = WorkerThread(self.transcription_queue, self.sd_prompt_queue, self.app_params, self.theme_input.text())
            self.worker.caption_updated.connect(self.update_caption)
            self.worker.progress_updated.connect(self.update_progress)
            
            self.img_update_worker = ImageUpdateThread(self.sd_prompt_queue, self.app_params)
            self.img_update_worker.image_updated.connect(self.update_image)
            self.img_update_worker.primary_pixmap_updated.connect(self.update_primary_pixmap)
            self.img_update_worker.depth_pixmap_updated.connect(self.update_depth_pixmap)

            self.worker.start()
            self.img_update_worker.start()
            self.start_button.setText("Stop")

        else:
            self.img_update_worker.stop()
            self.worker.stop()
            self.img_update_worker = None
            self.worker = None

            self.vad_worker.stop()
            self.whisper_worker.stop()

            #self.worker.terminate()
            self.start_button.setText("Start")
            
            self.sd_prompt_queue = None
            self.transcription_queue = None

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

    def update_depth_pixmap(self, pixmap):
        self.depth_pixmap = pixmap

        self.update_image_label()

    def update_primary_pixmap(self, pixmap):
        self.primary_pixmap = pixmap

        self.update_image_label()

    def update_image_label(self):
        if self.display_primary_img and self.primary_pixmap is not None:
            pixmap = self.primary_pixmap
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size()))
        elif not self.display_primary_img and self.depth_pixmap is not None:
            pixmap = self.depth_pixmap
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size()))

    def update_image(self, pixmap):
        print("not doing anything...")
        #pixmap = QPixmap.fromImage(image)
        #self.image_label.setPixmap(pixmap.scaled(self.image_label.size()))

    def swap_image(self):
        self.display_primary_img = (not self.display_primary_img)
        self.update_image_label()

    def update_caption(self, caption):
        self.caption_label.setText(caption)
        #self.log_widget.append(f"Caption updated: {caption}")

    def update_progress(self, value, label):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(label)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.vad_worker.stop()
            self.whisper_worker.stop()
            self.worker.stop()  # Gracefully stop the worker thread
            self.worker.wait()  # Wait for the thread to finish
        
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

    # creating the LLM pipeline

    print("Creating an llm pipeline to run on ", llm_device)

    llm_model_path = r"./models/llama-3.1-8b-instruct/INT4_compressed_weights"

    if llm_device == 'NPU':
        pipeline_config = {"MAX_PROMPT_LEN": 1536}
        llm_pipe = ov_genai.LLMPipeline(llm_model_path, llm_device, pipeline_config)
    else:
        llm_pipe = ov_genai.LLMPipeline(llm_model_path, llm_device)

    app_params["llm"] = llm_pipe

    print("Done creating our llm..")

    #print("Creating a stable diffusion pipeline to run on ", sd_device)
    app_params["sd_device"] = sd_device
    app_params["super_res_device"] = super_res_device

    #sd_pipe = ov_genai.Text2ImagePipeline(r"models/LCM_Dreamshaper_v7/FP16", sd_device)

    #app_params["sd"] = sd_pipe
    #print("done creating the stable diffusion pipeline...")

    app_params["whisper_device"] = whisper_device

    print("Initializing Super Res Model to run on ", super_res_device)
    #model_path_sr = Path(f"models/single-image-super-resolution-1033.xml")  # realesrgan.xml")
    #super_res_compiled_model, super_res_upsample_factor = superres_load(model_path_sr, super_res_device, h_custom=432,
    #                                                                    w_custom=768)
    #app_params["super_res_compiled_model"] = super_res_compiled_model
    #app_params["super_res_upsample_factor"] = super_res_upsample_factor
    #print("Initializing Super Res Model done...")

    #print("Initializing Depth Anything v2 model to run on ", depth_anything_device)
    #OV_DEPTH_ANYTHING_PATH = Path(f"models/depth_anything_v2_vits.xml")
    #depth_compiled_model = core.compile_model(OV_DEPTH_ANYTHING_PATH, device_name=depth_anything_device)
    #app_params["depth_compiled_model"] = depth_compiled_model
    #print("Initializing Depth Anything v2 done...")

    print("Demo is ready!")
    window = MainWindow(app_params)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()