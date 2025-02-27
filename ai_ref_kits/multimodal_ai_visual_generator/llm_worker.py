from async_worker import AsyncWorker
from queue import Empty   
from multiprocessing import Queue

# A async (subprocess) class that runs anything LLM-related (orchestrator, subworkers, etc.)
# They should all be invoked from within a single instance of this class, as we ideally want
# orchestrator and subagents to all share the same instance of a GenAI LLM Pipeline.
# (as each instance of an LLM Pipeline would need dedicated set of weights loaded into memory)
class LLMWorker(AsyncWorker):
    def __init__(self, transcription_queue, sd_prompt_queue, ui_update_queue, llm_device, theme):
        super().__init__()
        
        self.transcription_queue = transcription_queue
        self.sd_prompt_queue = sd_prompt_queue
        self.llm_device = llm_device
        self.theme = theme
        self.ui_update_queue = ui_update_queue
        
    def _work_loop(self):
        import openvino_genai as ov_genai
        
        print("Creating an llm pipeline to run on ", self.llm_device)
        
        llm_model_path = r"./models/llama-3.1-8b-instruct/INT4_compressed_weights"
        llm_device = self.llm_device
        
        if llm_device == 'NPU':
            pipeline_config = {"MAX_PROMPT_LEN": 1536}
            llm_pipe = ov_genai.LLMPipeline(llm_model_path, llm_device, pipeline_config)
        else:
            llm_pipe = ov_genai.LLMPipeline(llm_model_path, llm_device)
            
        from llm_processing.orchestrator import LLMOrchestrator
        
        orchestrator = LLMOrchestrator(llm_pipe, self.ui_update_queue, self.theme)

        while self._running.value:
            try:
                #self.progress_updated.emit(0, "listening")
                
                # Each queue entry is a tuple:
                # (start_time_in_seconds, segment_length_in_seconds, transcription)
                queue_entry = self.transcription_queue.get(timeout=1)
                transcription = queue_entry[2]
                transcription = f"\"{transcription}\""

                orchestrator_result = orchestrator.run(transcription=transcription)
                
                #Right now, it's either a string (to run SD), or it's None.
                # This will change to a dictionary once the orchestrator turns into
                # an actual orchestrator.
                if orchestrator_result is not None:
                    self.sd_prompt_queue.put(orchestrator_result)
                
            except Empty:
                continue  # Queue is empty, just wait
                
def test_main():
    sd_prompt_queue = Queue()
    transcription_queue = Queue()
    ui_update_queue = Queue()
    
    llm_worker = LLMWorker(transcription_queue, sd_prompt_queue, ui_update_queue, "GPU", "Medieval Fantasy Adventure")
    llm_worker.start()

    try:
        while True:
            try:
                transcription = input('')
            except EOFError:
                break

            transcription_queue.put((0, 1, transcription))
            
    except KeyboardInterrupt:
        print("Main: Stopping workers...")
        llm_worker.stop()
        
if __name__ == "__main__":
    test_main()