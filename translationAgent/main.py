import threading
import time
from queue import Queue

import nltk
import numpy as np
import sounddevice as sd
import torch
import whisper
from langchain.chains import ConversationChain
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from rich.console import Console
from transformers import AutoProcessor, BarkModel

SAMPLE_RATE = 24000  # 24kHz is the default sampling rate for Bark


class TextToSpeechService:
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sampling_rate: int = SAMPLE_RATE,
    ):
        self.device = device
        self.sampling_rate = sampling_rate
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.model = BarkModel.from_pretrained("suno/bark-small").to(self.device)

    def synthesize(
        self, text: str, voice_preset: str = "v2/en_speaker_6"
    ) -> tuple[np.ndarray, int]:
        inputs = self.processor(
            text, voice_preset=voice_preset, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            audio_array = self.model.generate(
                **inputs, pad_token_id=self.processor.tokenizer.eos_token_id
            )

        audio_array = audio_array.cpu().numpy().squeeze()
        return audio_array, self.sampling_rate

    def long_synthesize(
        self, text: str, voice_preset: str = "v2/en_speaker_6"
    ) -> tuple[np.ndarray, int]:
        pieces = []
        sentences = nltk.sent_tokenize(text)
        silence = np.zeros(int(0.25 * self.sampling_rate))

        for sent in sentences:
            audio_array, sample_rate = self.synthesize(sent, voice_preset=voice_preset)
            pieces += [audio_array, silence.copy()]

        return np.concatenate(pieces, axis=0), self.sampling_rate


def record_audio(stop_event: threading.Event, data_queue: Queue):
    def callback(input_data, frames, time, status):
        if status:
            console.print(f"[yellow]Recording status: {status}[/yellow]")
        data_queue.put(input_data.copy())

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            dtype=np.int16,
            channels=1,
            callback=callback,
            blocksize=1000,
        ):
            console.print("[green]Recording started...[/green]")
            stop_event.wait()
    except Exception as e:
        console.print(f"[red]Error in audio recording: {e}[/red]")
    finally:
        console.print("[yellow]Recording stopped.[/yellow]")


def transcribe_audio(stt: whisper.Whisper, audio_array: np.ndarray) -> str:
    result = stt.transcribe(audio_array, fp16=False)
    if not isinstance(result.get("text", None), str):
        raise ValueError("No text received")

    return result["text"].strip()


def get_llm_response(chain: ConversationChain, prompt: str) -> str:
    """Get a response from the LLM
    Args:
        chain (ConversationChain): The chain to use for the LLM
        prompt (str): The prompt to use for the LLM
    Returns:
        str: The generated response
    """
    output = chain.invoke(input=prompt)
    print(output.get("response", ""))
    if output.get("response", "").startswith("AI Language Tutor:"):
        response = output["response"][len("AI Language Tutor:") :].strip()
    else:
        response = output.get("response", "")

    return response


def play_audio(audio_array: np.ndarray, sample_rate: int) -> None:
    """Play audio
    Args:
        sample_rate (int): The sample rate of the audio
        audio_array (np.ndarray): The audio array to play
    """
    sd.play(audio_array, sample_rate, blocking=True)
    sd.wait()


def start_listening_event() -> tuple[threading.Event, Queue, threading.Thread]:
    console.input("Press Enter to start recording... and Ctrl+C to stop")
    data_queue = Queue()
    stop_event = threading.Event()
    recording_thread = threading.Thread(
        target=record_audio, args=(stop_event, data_queue)
    )
    recording_thread.start()
    return stop_event, data_queue, recording_thread


def stop_listening_event(
    data_queue: Queue, stop_event: threading.Event, recording_thread: threading.Thread
) -> np.ndarray:
    """Stop the recording thread and return the audio data
    Args:
        data_queue (Queue): The queue to get the audio data from
        stop_event (threading.Event): The event to stop the recording thread
        recording_thread (threading.Thread): The thread to stop
    Returns:
        np.ndarray: The audio data
    """
    stop_event.set()
    recording_thread.join()
    audio_data = np.concatenate(list(data_queue.queue), axis=0)
    return audio_data


def process_event(data_array: np.ndarray):
    """Process the audio data
    Args:
        data_array (np.ndarray): The audio data
    """
    if data_array is None or len(data_array) == 0:
        console.print(
            "[red]No audio data received. Ensure the microphone is working.[/red]"
        )

    if data_array.ndim > 0:
        data_array = data_array.flatten()

    audio_array = data_array.astype(np.float32) / np.iinfo(np.int16).max
    if audio_array.size > 0:
        with console.status("Transcribing...", spinner="dots"):
            text = transcribe_audio(stt, audio_array)
        console.print(f"[yellow]You: {text}[/yellow]")

        with console.status("Generating response...", spinner="dots"):
            response = get_llm_response(chain, text)
            audio_array, sample_rate = tts.long_synthesize(response)
        console.print(f"[green]AI Language Tutor: {response}[/green]")

        with console.status("Synthesizing response...", spinner="circle"):
            play_audio(audio_array, sample_rate)
    else:
        console.print(
            "[red]No audio data received. Ensure the microphone is working.[/red]"
        )


console = Console()
stt = whisper.load_model("base.en")
tts = TextToSpeechService()

template = """
You are a translator AI tutor. Your goal is to make sure you teach about foreigh languages. You are helpful and friendly. 
When you reply, you reply sharply and concisely.
The responses should be less than 300 characters. 
The conversation transcript is as follows:
{history}
And here is the user's follow-up: {input}
Your response:
"""
PROMPT = PromptTemplate(
    template=template,
    input_variables=["history", "input"],
)

chain = ConversationChain(
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="AI Language Tutor"),
    llm=Ollama(model="llama3.1"),
)


def listen_for_audio() -> np.ndarray:
    console.input("[green]Press Enter to start recording...[/green]")
    data_queue = Queue()
    stop_event = threading.Event()

    def record_audio():
        def callback(input_data, frames, time, status):
            if status:
                console.print(f"[yellow]Recording status: {status}[/yellow]")
            data_queue.put(input_data.copy())

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                dtype=np.int16,
                channels=1,
                callback=callback,
                blocksize=1000,
            ):
                console.print("[green]Recording started...[/green]")
                stop_event.wait()
        except Exception as e:
            console.print(f"[red]Error in audio recording: {e}[/red]")
        finally:
            console.print("[yellow]Recording stopped.[/yellow]")

    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()

    try:
        input("Press Enter to stop recording...")
    except KeyboardInterrupt:
        console.print("\n[red]Exiting...[/red]")
    finally:
        stop_event.set()
        recording_thread.join()

    audio_data = np.concatenate(list(data_queue.queue), axis=0)
    return audio_data


if __name__ == "__main__":
    console.print("Welcome to the AI Language Tutor. You can start by pressing Enter.")
    try:
        while True:
            audio_array = listen_for_audio()
            process_event(audio_array)
    except KeyboardInterrupt:
        console.print("\n[red]Exiting...[/red]")

    console.print("[blue]Thank you for using the AI Language Tutor![/blue]")
