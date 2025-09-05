"""
Módulo inicializador para video_processor
"""
from .video_processor import (
    VideoProcessor, 
    video_processor, 
    VideoChunk,
    VideoWriter,
    VideoDepthWriter,
    create_video_processor
)

__all__ = [
    'VideoProcessor', 
    'video_processor', 
    'VideoChunk',
    'VideoWriter',
    'VideoDepthWriter', 
    'create_video_processor'
]
