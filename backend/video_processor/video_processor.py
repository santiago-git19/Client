import os
import threading
import time
import cv2
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Callable, Union
from dataclasses import dataclass

from ..config.settings import SystemConfig
from ..camera_manager import camera_manager


@dataclass
class VideoChunk:
    """Información de un chunk de video"""
    chunk_id: str
    camera_id: int
    session_id: str
    patient_id: str
    sequence_number: int
    color_file_path: str
    duration_seconds: float
    timestamp: datetime
    color_file_size_bytes: int
    # Campos opcionales para compatibilidad con VideoWriter
    depth_file_path: Optional[str] = None
    depth_file_size_bytes: Optional[int] = None
    # Para compatibilidad con VideoWriter (que solo tiene un archivo)
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None


class VideoWriter:
    """Manejador de escritura de video para una cámara"""
    
    def __init__(self, camera_id: int, output_path: str, config=None):
        self.camera_id = camera_id
        self.output_path = output_path
        self.config = config  # Almacenar configuración (opcional)
        self.writer: Optional[cv2.VideoWriter] = None
        self.frame_count = 0
        self.start_time: Optional[datetime] = None
        
    def initialize(self, frame_width: int, frame_height: int, fps: int) -> bool:
        """Inicializar el writer de video"""
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                fps,
                (frame_width, frame_height)
            )
            
            if not self.writer.isOpened():
                print(f"Error: No se pudo crear el video writer para cámara {self.camera_id}")
                return False
                
            self.start_time = datetime.now()
            print(f"Video writer inicializado para cámara {self.camera_id}: {self.output_path}")
            return True
            
        except Exception as e:
            print(f"Error inicializando video writer para cámara {self.camera_id}: {e}")
            return False
    
    def write_frame(self, frame) -> bool:
        """Escribir un frame al video"""
        if self.writer is None or not self.writer.isOpened():
            return False
            
        try:
            self.writer.write(frame)
            self.frame_count += 1
            return True
        except Exception as e:
            print(f"Error escribiendo frame en cámara {self.camera_id}: {e}")
            return False
    
    def finalize(self) -> Optional[VideoChunk]:
        """Finalizar el video y retornar información del chunk"""
        if self.writer is None:
            return None
            
        try:
            self.writer.release()
            self.writer = None
            
            # Verificar que el archivo se creó correctamente
            if not os.path.exists(self.output_path):
                print(f"Error: Archivo de video no encontrado: {self.output_path}")
                return None
            
            file_size = os.path.getsize(self.output_path)
            duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            
            # Generar información del chunk
            chunk_info = VideoChunk(
                chunk_id=str(uuid.uuid4()),
                camera_id=self.camera_id,
                session_id="",  # Se asignará externamente
                patient_id="",  # Se asignará externamente
                sequence_number=0,  # Se asignará externamente
                color_file_path=self.output_path,  # VideoWriter solo maneja color
                duration_seconds=duration,
                timestamp=self.start_time or datetime.now(),
                color_file_size_bytes=file_size,
                # Para compatibilidad hacia atrás
                file_path=self.output_path,
                file_size_bytes=file_size
            )
            
            print(f"Chunk finalizado para cámara {self.camera_id}: {file_size} bytes, {duration:.2f}s")
            return chunk_info
            
        except Exception as e:
            print(f"Error finalizando video para cámara {self.camera_id}: {e}")
            return None


class VideoDepthWriter:
    """Manejador de escritura de video de color y datos de profundidad para una cámara"""
    
    def __init__(self, camera_id: int, output_path: str, chunk_id: str, config=None):
        self.camera_id = camera_id
        self.output_path = output_path
        self.chunk_id = chunk_id
        self.config = config
        
        # Paths para archivos de color y profundidad
        self.color_path = os.path.join(output_path, f"{chunk_id}_color.mp4")
        self.depth_path = os.path.join(output_path, f"{chunk_id}_depth.npy")
        
        self.color_writer: Optional[cv2.VideoWriter] = None
        self.depth_frames: List[np.ndarray] = []
        self.frame_count = 0
        self.start_time: Optional[datetime] = None
        
    def initialize(self, frame_width: int, frame_height: int, fps: int) -> bool:
        """Inicializar el writer de video de color"""
        try:
            # Crear directorio si no existe
            os.makedirs(self.output_path, exist_ok=True)
            
            # Inicializar writer de color
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.color_writer = cv2.VideoWriter(
                self.color_path,
                fourcc,
                fps,
                (frame_width, frame_height)
            )
            
            if not self.color_writer.isOpened():
                print(f"Error: No se pudo crear el video writer para cámara {self.camera_id}")
                return False
                
            self.start_time = datetime.now()
            print(f"Video+Depth writer inicializado para cámara {self.camera_id}")
            print(f"  Color: {self.color_path}")
            print(f"  Depth: {self.depth_path}")
            return True
            
        except Exception as e:
            print(f"Error inicializando video+depth writer para cámara {self.camera_id}: {e}")
            return False
    
    def write_frames(self, color_frame: np.ndarray, depth_frame: np.ndarray) -> bool:
        """Escribir frames de color y profundidad"""
        if self.color_writer is None or not self.color_writer.isOpened():
            return False
            
        try:
            # Escribir frame de color
            self.color_writer.write(color_frame)
            
            # Almacenar frame de profundidad
            self.depth_frames.append(depth_frame.copy())
            
            self.frame_count += 1
            return True
        except Exception as e:
            print(f"Error escribiendo frames en cámara {self.camera_id}: {e}")
            return False
    
    def finalize(self) -> Optional[VideoChunk]:
        """Finalizar el video y guardar datos de profundidad"""
        if self.color_writer is None:
            return None
            
        try:
            # Finalizar video de color
            self.color_writer.release()
            self.color_writer = None
            
            # Guardar datos de profundidad como numpy array
            if self.depth_frames:
                depth_array = np.array(self.depth_frames)
                np.save(self.depth_path, depth_array)
            
            # Verificar que los archivos se crearon correctamente
            if not os.path.exists(self.color_path):
                print(f"Error: Archivo de video color no encontrado: {self.color_path}")
                return None
                
            if not os.path.exists(self.depth_path):
                print(f"Error: Archivo de profundidad no encontrado: {self.depth_path}")
                return None
            
            color_size = os.path.getsize(self.color_path)
            depth_size = os.path.getsize(self.depth_path)
            duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            
            # Generar información del chunk
            chunk_info = VideoChunk(
                chunk_id=self.chunk_id,
                camera_id=self.camera_id,
                session_id="",  # Se asignará externamente
                patient_id="",  # Se asignará externamente
                sequence_number=0,  # Se asignará externamente
                color_file_path=self.color_path,
                depth_file_path=self.depth_path,
                duration_seconds=duration,
                timestamp=self.start_time or datetime.now(),
                color_file_size_bytes=color_size,
                depth_file_size_bytes=depth_size
            )
            
            print(f"Chunk finalizado - Cámara {self.camera_id}:")
            print(f"  Color: {color_size} bytes ({self.frame_count} frames)")
            print(f"  Depth: {depth_size} bytes ({len(self.depth_frames)} frames)")
            print(f"  Duración: {duration:.2f}s")
            
            return chunk_info
            
        except Exception as e:
            print(f"Error finalizando chunk para cámara {self.camera_id}: {e}")
            return None


class VideoProcessor:
    """Procesador principal de video multi-cámara con color y profundidad"""
    
    def __init__(self, use_depth: bool = True):
        self.recording_active = False
        self.session_id: Optional[str] = None
        self.patient_id: Optional[str] = None
        self.use_depth = use_depth  # Determina si usar VideoDepthWriter o VideoWriter
        self.current_writers: Dict[int, Union[VideoWriter, VideoDepthWriter]] = {}
        self.chunk_sequence: Dict[int, int] = {}  # Indica, para cada cámara (identificada por el índice del diccionario), el número de secuencia del chunk que se está grabando
        self.recording_thread: Optional[threading.Thread] = None
        self.upload_callbacks: List[Callable[[VideoChunk], None]] = []
        
        # Configuración
        self.config = SystemConfig.RECORDING
        
    def start_session(self, patient_id: str, session_id: str = "1") -> str: # Se emplea en el start_recording del app.py
        """Iniciar nueva sesión de grabación"""
        if self.recording_active:
            raise Exception("Ya hay una sesión activa")
            
        self.session_id = session_id  # Usar el session_id proporcionado
        self.patient_id = patient_id
        self.chunk_sequence.clear()
        
        # Limpiar directorios de cámaras existentes
        self._cleanup_camera_directories()
        
        # Inicializar secuencias para cada cámara empezando en 0
        for camera_id in camera_manager.cameras:
            self.chunk_sequence[camera_id] = 0
            
        print(f"Nueva sesión iniciada: {self.session_id} para paciente: {self.patient_id}")
        return self.session_id
    
    def start_recording(self) -> bool:
        """Iniciar grabación con chunks automáticos"""
        if self.recording_active:
            return False
            
        if not self.session_id:
            raise Exception("No hay sesión activa. Llamar start_session() primero")
            
        try:
            self.recording_active = True
            
            # Iniciar grabación en cámaras
            if not camera_manager.start_recording_all(session_id=self.session_id, patient_id=self.patient_id):
                self.recording_active = False
                return False
            
            # Iniciar hilo de grabación por chunks
            self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
            self.recording_thread.start()
            
            print(f"Grabación iniciada para sesión: {self.session_id}", flush=True)
            return True
            
        except Exception as e:
            print(f"Error iniciando grabación: {e}")
            self.recording_active = False
            return False
    
    def stop_recording(self) -> List[VideoChunk]:
        """Detener grabación y finalizar chunks pendientes"""
        if not self.recording_active:
            return []
            
        print("Deteniendo grabación...")
        print("Generando chunks finales con frames restantes...")
        
        # Marcar que debe detenerse la grabación, pero permitir que termine el chunk actual
        self.recording_active = False
        
        # Esperar a que termine el hilo de grabación
        if self.recording_thread and self.recording_thread.is_alive():
            print("Esperando a que termine el hilo de grabación...")
            self.recording_thread.join(timeout=15)  # Aumentar timeout para permitir finalización
        
        # Generar chunks finales con cualquier frame restante
        final_chunks = []
        
        # Capturar algunos frames adicionales para el chunk final si hay writers activos
        if self.current_writers:
            print(f"Capturando frames finales para {len(self.current_writers)} cámaras...")
            
            # Capturar hasta 1 segundo adicional de frames para el chunk final
            frames_captured = 0
            max_final_frames = 30  # Aproximadamente 1 segundo a 30fps
            
            try:
                for _ in range(max_final_frames):
                    frames_written_this_cycle = 0
                    
                    for camera_id in list(self.current_writers.keys()):
                        writer = self.current_writers[camera_id]
                        
                        # Verificar qué tipo de writer es
                        if isinstance(writer, VideoDepthWriter):
                            # Para VideoDepthWriter necesitamos color y profundidad
                            color_frame = camera_manager.get_frame(camera_id)
                            depth_frame = camera_manager.get_depth_frame(camera_id)
                            
                            if color_frame is not None and depth_frame is not None and camera_id in self.current_writers:
                                if writer.write_frames(color_frame, depth_frame):
                                    frames_written_this_cycle += 1
                            elif color_frame is None or depth_frame is None:
                                if frames_captured % 30 == 0:  # Log cada segundo aproximadamente
                                    missing = []
                                    if color_frame is None:
                                        missing.append("color")
                                    if depth_frame is None:
                                        missing.append("depth")
                                    print(f"Cámara {camera_id}: No se pudo obtener frame de {'/'.join(missing)}")
                        
                        elif isinstance(writer, VideoWriter):
                            # Para VideoWriter solo necesitamos el frame de color
                            color_frame = camera_manager.get_frame(camera_id)
                            if color_frame is not None and camera_id in self.current_writers:
                                if writer.write_frame(color_frame):
                                    frames_written_this_cycle += 1
                        
                        else:
                            print(f"Tipo de writer desconocido para cámara {camera_id}: {type(writer)}")
                    
                    if frames_written_this_cycle == 0:
                        break  # No hay más frames disponibles
                    
                    frames_captured += frames_written_this_cycle
                    time.sleep(1/30)  # Aproximadamente 30fps
                
                if frames_captured > 0:
                    print(f"Capturados {frames_captured} frames adicionales para chunks finales")
                
            except Exception as e:
                print(f"Error capturando frames finales: {e}")
        
        # Finalizar writers actuales
        print("Finalizando writers actuales...")
        for camera_id, writer in self.current_writers.items():
            chunk = self._finalize_writer(camera_id, writer)
            if chunk:
                final_chunks.append(chunk)
                print(f"Chunk final generado para cámara {camera_id}: {chunk.duration_seconds:.2f}s")
        
        self.current_writers.clear()
        camera_manager.stop_recording_all()
        
        print(f"Grabación detenida. {len(final_chunks)} chunks finales generados")
        return final_chunks
    
    def cancel_recording(self) -> bool:
        """Cancelar grabación y limpiar archivos"""
        if not self.recording_active:
            return True
            
        print("Cancelando grabación...")
        self.recording_active = False
        
        # Esperar a que termine el hilo
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=10)
        
        # Cerrar writers y eliminar archivos
        for camera_id, writer in self.current_writers.items():
            try:
                if isinstance(writer, VideoDepthWriter):
                    # Para VideoDepthWriter
                    if writer.color_writer:
                        writer.color_writer.release()
                    
                    # Eliminar archivos de color y profundidad
                    if os.path.exists(writer.color_path):
                        os.remove(writer.color_path)
                        print(f"Archivo de color eliminado: {writer.color_path}")
                    if os.path.exists(writer.depth_path):
                        os.remove(writer.depth_path)
                        print(f"Archivo de profundidad eliminado: {writer.depth_path}")
                        
                elif isinstance(writer, VideoWriter):
                    # Para VideoWriter
                    if writer.writer:
                        writer.writer.release()
                        time.sleep(0.5)  # Esperar un momento para asegurar que el archivo se cierre correctamente
                    if os.path.exists(writer.output_path):
                        os.remove(writer.output_path)
                        print(f"Archivo eliminado: {writer.output_path}")
                        
            except Exception as e:
                print(f"Error eliminando archivos de cámara {camera_id}: {e}")
        
        self.current_writers.clear()
        camera_manager.stop_recording_all()
        
        # Limpiar directorio temporal de la sesión
        self._cleanup_session_files()
        
        print("Grabación cancelada y archivos limpiados")
        return True
    
    def _recording_loop(self):
        """Bucle principal de grabación"""
        try:
            print("Iniciando bucle de grabación...", flush=True)
            while self.recording_active:
                print(f"Nuevo ciclo de grabación - cámaras disponibles: {list(camera_manager.cameras.keys())}")

                # Crear nuevos writers si es necesario
                self._create_new_writers()
                
                start_time = time.time()
                frames_written = {camera_id: 0 for camera_id in camera_manager.cameras} # Cada id de cámara tendrá asignado el cero al principio, y conforme cree frames, se irán aumentando
                
                # Grabar durante la duración del chunk
                print(f"Iniciando grabación de chunk de {self.config.chunk_duration_seconds} segundos...")
                frame_count = 0
                while (time.time() - start_time) < self.config.chunk_duration_seconds and self.recording_active:
                    # Capturar frames de todas las cámaras (sincronización por software)
                    
                    for camera_id in camera_manager.cameras:
                        if camera_id not in self.current_writers:
                            continue
                            
                        writer = self.current_writers[camera_id]
                        
                        if isinstance(writer, VideoDepthWriter):
                            # Para VideoDepthWriter necesitamos color y profundidad
                            color_frame = camera_manager.get_frame(camera_id)
                            depth_frame = camera_manager.get_depth_frame(camera_id)
                            
                            if color_frame is not None and depth_frame is not None:
                                if writer.write_frames(color_frame, depth_frame):
                                    frames_written[camera_id] += 1
                            elif frame_count % 30 == 0:  # Log cada segundo aproximadamente
                                missing = []
                                if color_frame is None:
                                    missing.append("color")
                                if depth_frame is None:
                                    missing.append("depth")
                                print(f"Cámara {camera_id}: No se pudo obtener frame de {'/'.join(missing)}")
                                
                        elif isinstance(writer, VideoWriter):
                            # Para VideoWriter solo necesitamos color
                            color_frame = camera_manager.get_frame(camera_id)
                            
                            if color_frame is not None:
                                if writer.write_frame(color_frame):
                                    frames_written[camera_id] += 1
                            elif frame_count % 30 == 0:
                                print(f"Cámara {camera_id}: No se pudo obtener frame de color")
                                
                        else:
                            if frame_count % 30 == 0:
                                print(f"Tipo de writer desconocido para cámara {camera_id}: {type(writer)}")

                    frame_count += 1
                
                elapsed = time.time() - start_time
                print(f"Chunk completado en {elapsed:.2f}s - Frames escritos por cámara: {frames_written}")
                
                # Finalizar chunk actual y crear el siguiente
                if self.recording_active:
                    print("Finalizando chunks actuales...")
                    self._finalize_current_chunks()
                    print(f"Estado después de finalizar - chunk_sequence: {self.chunk_sequence}")
                else:
                    print("Grabación detenida, no se creará siguiente chunk")

        except Exception as e:
            print(f"Error en bucle de grabación: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Bucle de grabación terminado")
            self.recording_active = False
    
    def _create_new_writers(self):
        """Crear nuevos writers para el siguiente chunk"""
        print(f"Creando writers para cámaras: {list(camera_manager.cameras.keys())}")
        
        for camera_id in camera_manager.cameras:
            if camera_id not in self.current_writers:
                # Generar ID único para el chunk
                chunk_id = f"cam{camera_id}_seq{self.chunk_sequence.get(camera_id, 0)}_{int(time.time())}"
                
                # Crear directorio para esta cámara
                camera_dir = os.path.join(SystemConfig.TEMP_VIDEO_DIR, f"camera{camera_id}")
                
                # Obtener un frame para determinar dimensiones
                frame = camera_manager.get_frame(camera_id)
                if frame is not None:
                    height, width = frame.shape[:2]
                    # Obtener FPS real de la cámara
                    fps = camera_manager.cameras[camera_id].get_real_fps()
                    
                    if self.use_depth:
                        # Crear VideoDepthWriter para color + profundidad
                        print(f"Generando VideoDepthWriter para cámara {camera_id}: {chunk_id}")
                        writer = VideoDepthWriter(camera_id, camera_dir, chunk_id, self.config)
                        writer_type = "VideoDepthWriter"
                    else:
                        # Crear VideoWriter para solo color
                        print(f"Generando VideoWriter para cámara {camera_id}: {chunk_id}")
                        output_path = os.path.join(camera_dir, f"{chunk_id}.mp4")
                        os.makedirs(camera_dir, exist_ok=True)
                        writer = VideoWriter(camera_id, output_path, self.config)
                        writer_type = "VideoWriter"
                    
                    print(f"Inicializando {writer_type} para cámara {camera_id}: {width}x{height}@{fps}fps")
                    
                    if writer.initialize(width, height, fps):
                        self.current_writers[camera_id] = writer
                        print(f"{writer_type} creado exitosamente para cámara {camera_id}")
                    else:
                        print(f"Error inicializando {writer_type} para cámara {camera_id}")
                else:
                    print(f"No se pudo obtener frame de prueba para cámara {camera_id}")
        
        print(f"Writers activos: {list(self.current_writers.keys())}")
    
    def _finalize_current_chunks(self):
        """Finalizar chunks actuales y enviarlos"""
        chunks_to_upload = []
        
        for camera_id, writer in list(self.current_writers.items()):
            chunk = self._finalize_writer(camera_id, writer)
            if chunk:
                chunks_to_upload.append(chunk)
        
        self.current_writers.clear()
        
        # Enviar chunks en paralelo
        for chunk in chunks_to_upload:
            threading.Thread(target=self._upload_chunk, args=(chunk,), daemon=True).start()
    
    def _finalize_writer(self, camera_id: int, writer) -> Optional[VideoChunk]:
        """Finalizar un writer específico (compatible con VideoWriter y VideoDepthWriter)"""
        if not isinstance(writer, (VideoWriter, VideoDepthWriter)):
            print(f"Tipo de writer no soportado para cámara {camera_id}: {type(writer)}")
            return None
            
        chunk = writer.finalize()
        if chunk:
            chunk.session_id = self.session_id
            chunk.patient_id = self.patient_id
            chunk.sequence_number = self.chunk_sequence[camera_id]
            # Incrementar DESPUÉS de asignar el número al chunk
            self.chunk_sequence[camera_id] += 1
        
        return chunk
    
    def _upload_chunk(self, chunk: VideoChunk):
        """Subir chunk al servidor (placeholder)"""
        try:
            # Llamar callbacks registrados
            for callback in self.upload_callbacks:
                callback(chunk)
                
            print(f"Chunk enviado: Cámara {chunk.camera_id}, Secuencia {chunk.sequence_number}")
            
        except Exception as e:
            print(f"Error enviando chunk: {e}")
    
    def _generate_chunk_path(self, camera_id: int) -> str:
        """Generar ruta para un nuevo chunk"""
        camera_dir = os.path.join(SystemConfig.TEMP_VIDEO_DIR, f"camera{camera_id}")
        
        # Crear directorio de cámara si no existe
        os.makedirs(camera_dir, exist_ok=True)
        
        # Obtener el número de secuencia para esta cámara
        # Asegurar que la cámara tenga una entrada en chunk_sequence
        if camera_id not in self.chunk_sequence:
            self.chunk_sequence[camera_id] = 0
        
        sequence_number = self.chunk_sequence[camera_id]
        filename = f"{sequence_number}.mp4"
        
        print(f"Generando chunk para cámara {camera_id}: secuencia {sequence_number} → {filename}")
        
        return os.path.join(camera_dir, filename)
    
    def _cleanup_session_files(self):
        """Limpiar archivos de las cámaras"""
        try:
            # Limpiar archivos de cada cámara en lugar de por sesión
            for camera_id in self.chunk_sequence.keys():
                camera_dir = os.path.join(SystemConfig.TEMP_VIDEO_DIR, f"camera{camera_id}")
                if os.path.exists(camera_dir):
                    import shutil
                    shutil.rmtree(camera_dir)
                    print(f"Directorio de cámara {camera_id} eliminado: {camera_dir}")
        except Exception as e:
            print(f"Error eliminando directorios de cámaras: {e}")
    
    def _cleanup_camera_directories(self): # Se emplea en start_session de VideoProcessor
        """Limpiar todos los directorios de cámaras existentes"""
        try:
            import shutil
            # Buscar y eliminar todos los directorios camera0, camera1, camera2, etc.
            for i in range(SystemConfig.MAX_CAMERAS):
                camera_dir = os.path.join(SystemConfig.TEMP_VIDEO_DIR, f"camera{i}")
                if os.path.exists(camera_dir):
                    shutil.rmtree(camera_dir)
                    print(f"Directorio existente eliminado: camera{i}")
        except Exception as e:
            print(f"Error limpiando directorios de cámaras: {e}")
    
    def add_upload_callback(self, callback: Callable[[VideoChunk], None]):
        """Añadir callback para cuando se genere un chunk"""
        self.upload_callbacks.append(callback)

    def cancel_current_session(self) -> bool: # Se emplea en upload_chunk_to_server de app.py
        """Cancelar la sesión actual completamente"""
        try:
            print("Cancelando sesión actual por fallo de cámaras...")
            
            # Cancelar grabación si está activa
            if self.recording_active:
                self.cancel_recording()
            
            # Limpiar estado de la sesión
            self.session_id = None
            self.patient_id = None
            self.chunk_sequence.clear()
            self.current_writers.clear()
            
            print("Sesión cancelada completamente")
            return True
            
        except Exception as e:
            print(f"Error cancelando sesión: {e}")
            return False


# Función factory para crear procesadores configurables
def create_video_processor(use_depth: bool = True) -> VideoProcessor:
    """
    Crear una instancia del procesador de video
    
    Args:
        use_depth: Si True, usa VideoDepthWriter (color + profundidad)
                   Si False, usa VideoWriter (solo color)
    """
    return VideoProcessor(use_depth=use_depth)

# Singleton del procesador de video con profundidad habilitada por defecto
video_processor = VideoProcessor(use_depth=True)
