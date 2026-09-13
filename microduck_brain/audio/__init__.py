"""Microduck Robot Brain Audio & Voice Response Package."""
from microduck_brain.audio.personality import Personality
from microduck_brain.audio.recipes import (
    greet,
    inquire,
    peck,
    chirp,
    coo,
    wheee,
    alarm,
)
from microduck_brain.audio.voice_interface import synthesize_human_command, generate_mission_audio_track

__all__ = [
    "Personality",
    "greet",
    "inquire",
    "peck",
    "chirp",
    "coo",
    "wheee",
    "alarm",
    "synthesize_human_command",
    "generate_mission_audio_track",
]
