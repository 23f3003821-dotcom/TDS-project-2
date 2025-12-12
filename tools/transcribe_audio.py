import os
import requests
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def transcribe_audio(audio_url: str) -> dict:
    """
    Downloads an audio file and transcribes it using OpenAI Whisper via AI Pipe.
    
    Parameters
    ----------
    audio_url : str
        URL of the audio file to transcribe.
    
    Returns
    -------
    dict
        {
            "transcription": <transcribed text>,
            "error": <error message if any>
        }
    """
    try:
        # Download the audio file
        audio_response = requests.get(audio_url)
        audio_response.raise_for_status()
        
        # Save temporarily
        temp_path = "/tmp/audio_temp.mp3"
        with open(temp_path, "wb") as f:
            f.write(audio_response.content)
        
        # Use OpenAI Whisper via AI Pipe
        api_key = os.getenv("API_KEY")
        
        with open(temp_path, "rb") as audio_file:
            response = requests.post(
                "https://aipipe.org/openrouter/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": audio_file},
                data={"model": "openai/whisper-1"}
            )
        
        if response.status_code == 200:
            result = response.json()
            return {"transcription": result.get("text", ""), "error": None}
        else:
            return {"transcription": "", "error": f"API Error: {response.status_code} - {response.text}"}
            
    except Exception as e:
        return {"transcription": "", "error": str(e)}
