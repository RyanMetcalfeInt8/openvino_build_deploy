from async_worker import AsyncWorker
from queue import Empty   
from multiprocessing import Queue

from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from superres import superres_load

def convert_result_to_image(result) -> np.ndarray:
    """
    Convert network result of floating point numbers to image with integer
    values from 0-255. Values outside this range are clipped to 0 and 255.

    :param result: a single superresolution network result in N,C,H,W shape
    """
    result = result.squeeze(0).transpose(1, 2, 0)
    result *= 255
    result[result < 0] = 0
    result[result > 255] = 255
    result = result.astype(np.uint8)
    return Image.fromarray(result)
    
class StableDiffusionWorker(AsyncWorker):
    def __init__(self, sd_prompt_queue, sd_device, super_res_device):
        super().__init__()
        
        self.result_img_queue = Queue()
        self.sd_device = sd_device
        self.sd_prompt_queue = sd_prompt_queue
        self.super_res_device = super_res_device
        
    def _work_loop(self):
        import openvino_genai as ov_genai
        
        print("Creating a stable diffusion pipeline to run on ", self.sd_device)
        sd_pipe = ov_genai.Text2ImagePipeline(r"models/LCM_Dreamshaper_v7/FP16", self.sd_device)
        
        print("Initializing Super Res Model to run on ", self.super_res_device)
        model_path_sr = Path(f"models/single-image-super-resolution-1033.xml")
        self.compiled_model, self.upsample_factor = superres_load(model_path_sr, self.super_res_device, 
                                                                  h_custom=432, w_custom=768)

        while self._running.value:
            try:
                prompt = self.sd_prompt_queue.get(timeout=1)
                
                print("prompt = ", prompt)
                import time
                t0 = time.time()
                image_tensor = sd_pipe.generate(
                    prompt,
                    width=768,
                    height=432,
                    num_inference_steps=5,
                    num_images_per_prompt=1)
                
                                
                sd_output = Image.fromarray(image_tensor.data[0])
                t1 = time.time()  
                sr_output = self.run_sr(np.array(sd_output))
                t2 = time.time()
                print("SD time = ", t1 - t0)
                print("SR time = ", t2 - t1)
                
                self.result_img_queue.put(sr_output)
                
            except Empty:
                continue  # Queue is empty, just wait
                
    def run_sr(self, img):
        input_image_original = np.expand_dims(img.transpose(2, 0, 1), axis=0)
        bicubic_image = cv2.resize(
        src=img, dsize=(768*self.upsample_factor, 432*self.upsample_factor), interpolation=cv2.INTER_CUBIC)
        input_image_bicubic = np.expand_dims(bicubic_image.transpose(2, 0, 1), axis=0)

        original_image_key, bicubic_image_key = self.compiled_model.inputs
        output_key = self.compiled_model.output(0)

        result = self.compiled_model(
        {
            original_image_key.any_name: input_image_original,
            bicubic_image_key.any_name: input_image_bicubic,
        }
        )[output_key]

        result_image = convert_result_to_image(result)

        return result_image
                
def test_main():
    sd_prompt_queue = Queue()
    sd_worker = StableDiffusionWorker(sd_prompt_queue, "GPU", "GPU")
    sd_worker.start()
    result_img_queue = sd_worker.result_img_queue
    
    try:
        # Just a loop to allow the workers to do their thing, and wait for user
        # to Ctrl-C
        while True:
            import time
            time.sleep(1)
            #print("got result!")
            sd_prompt_queue.put("A large castle!")
            img = result_img_queue.get()
            print(type(img))
    except KeyboardInterrupt:
        print("Main: Stopping workers...")
        sd_worker.stop()
        
if __name__ == "__main__":
    test_main()

        