import os
import time
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime

from ..camera_manager import camera_manager
from ..video_processor import video_processor, VideoChunk
from ..config.settings import SystemConfig, CameraConfig

# Variable global para rastrear cancelaciones por fallo de cámaras
camera_failure_detected = False


def create_app() -> Flask:
    # Ajustar la ruta para que apunte a la carpeta 'frontend' en el directorio raíz
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'frontend'))
    app = Flask(__name__, static_folder=frontend_dir)
    CORS(app)  # Permitir requests desde el frontend
    
    # Configuración
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
    
    # RUTAS PARA SERVIR EL FRONTEND
    @app.route('/')
    def index():
        """Servir el archivo principal del frontend"""
        return send_from_directory(app.static_folder, 'index.html')

    @app.route('/<path:path>')
    def serve_static(path):
        """Servir otros archivos estáticos (JS, CSS, etc.)"""
        return send_from_directory(app.static_folder, path)

    # Callback para envío de chunks al servidor
    def upload_chunk_to_server(chunk: VideoChunk):
        """Enviar chunk al servidor de procesamiento"""
        try:
            url = f"{SystemConfig.SERVER.base_url}{SystemConfig.SERVER.upload_endpoint}" 
            
            # Preparar archivos del chunk - compatible con VideoWriter y VideoDepthWriter
            files = {}
            files_to_cleanup = []
            
            # Archivo de color (siempre presente)
            if chunk.color_file_path and os.path.exists(chunk.color_file_path):
                files['file_color'] = open(chunk.color_file_path, 'rb')
                files_to_cleanup.append(chunk.color_file_path)
                print(f"Color file: {chunk.color_file_path}")
            elif chunk.file_path and os.path.exists(chunk.file_path):
                # Compatibilidad con VideoWriter (solo un archivo)
                files['file_color'] = open(chunk.file_path, 'rb')
                files_to_cleanup.append(chunk.file_path)
                print(f"Video file (color): {chunk.file_path}")
            
            # Archivo de profundidad (opcional - solo con VideoDepthWriter)
            if chunk.depth_file_path and os.path.exists(chunk.depth_file_path):
                files['file_depth'] = open(chunk.depth_file_path, 'rb')
                files_to_cleanup.append(chunk.depth_file_path)
                print(f"Depth file: {chunk.depth_file_path}")
            
            if not files:
                print(f"No hay archivos válidos para enviar en chunk {chunk.chunk_id}")
                return
            
            data = {
                'chunk_id': chunk.chunk_id,
                'camera_id': chunk.camera_id,
                'session_id': chunk.session_id,
                'patient_id': chunk.patient_id,
                'chunk_number': chunk.sequence_number,  # Server espera chunk_number
                'duration_seconds': chunk.duration_seconds,
                'timestamp': chunk.timestamp.isoformat(),
                'color_file_size_bytes': chunk.color_file_size_bytes or chunk.file_size_bytes or 0,
                'depth_file_size_bytes': chunk.depth_file_size_bytes or 0,
                'has_depth': chunk.depth_file_path is not None  # Indicar si tiene profundidad
            }
            
            response = requests.post(url, files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                print(f"Chunk enviado exitosamente: {chunk.chunk_id}")
                # Eliminar archivos locales después del envío exitoso
                try:
                    for file_path in files_to_cleanup:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            print(f"Archivo eliminado: {file_path}")
                except Exception as e:
                    print(f"Error eliminando archivos locales: {e}")
            elif response.status_code == 500:
                # Verificar si es un error de fallo de cámaras
                try:
                    error_data = response.json()
                    if error_data.get('error') == 'CAMERA_FAILURE_DETECTED':
                        print(f"FALLO DE CÁMARAS DETECTADO POR EL SERVIDOR")
                        print(f"Mensaje: {error_data.get('message', 'Error de cámaras')}")
                        print(f"Acción requerida: {error_data.get('action_required', 'Reiniciar switch')}")
                        
                        # Marcar que hubo un fallo de cámaras
                        global camera_failure_detected
                        camera_failure_detected = True
                        
                        # Cancelar la sesión actual inmediatamente
                        try:
                            print("Cancelando sesión local debido a fallo de cámaras...")
                            video_processor.cancel_current_session()
                            print("Sesión local cancelada por fallo de cámaras")
                        except Exception as cancel_error:
                            print(f"Error cancelando sesión local: {cancel_error}")
                        
                        return  # No continuar procesando este chunk
                except:
                    pass  # Si no se puede parsear como JSON, continuar con el manejo normal
                    
                print(f"Error 500 enviando chunk: {response.status_code} - {response.text}")
            else:
                print(f"Error enviando chunk: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error en upload_chunk_to_server: {e}")
        finally:
            # Cerrar todos los archivos abiertos
            for file_obj in files.values():
                try:
                    file_obj.close()
                except Exception:
                    pass
    
    # Registrar callback
    video_processor.add_upload_callback(upload_chunk_to_server)
    
    # ENDPOINTS DE CÁMARAS
    
    @app.route('/api/cameras/discover', methods=['GET'])
    def discover_cameras():
        """Dice cuántas cámaras hay conectadas"""
        try:
            cameras = camera_manager.discover_cameras()
            return jsonify({
                'success': True,
                'cameras': [
                    {
                        'camera_id': cam.camera_id,
                        'serial_number': cam.serial_number,
                        'is_connected': cam.is_connected
                    }
                    for cam in cameras
                ],
                'total_cameras': len(cameras)
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/cameras/initialize', methods=['POST'])
    def initialize_cameras():
        """Inicializar cámaras para grabación"""
        try:
            data = request.get_json() or {}
            camera_ids = data.get('camera_ids', [])
            
            if not camera_ids:
                # Descubrir e inicializar todas las cámaras disponibles
                discovered = camera_manager.discover_cameras()
                camera_ids = [cam.camera_id for cam in discovered]
            
            initialized = []
            errors = []
            
            for camera_id in camera_ids:
                # Crear nueva configuración para cada cámara
                config = CameraConfig(
                    camera_id=camera_id,
                    resolution_width=SystemConfig.DEFAULT_CAMERA_CONFIG.resolution_width,
                    resolution_height=SystemConfig.DEFAULT_CAMERA_CONFIG.resolution_height,
                    fps=SystemConfig.DEFAULT_CAMERA_CONFIG.fps,
                    format=SystemConfig.DEFAULT_CAMERA_CONFIG.format
                )
                
                if camera_manager.initialize_camera(camera_id, config):
                    initialized.append(camera_id)
                else:
                    errors.append(f"Error inicializando cámara {camera_id}")
            
            return jsonify({
                'success': len(errors) == 0,
                'initialized_cameras': initialized,
                'errors': errors,
                'total_initialized': len(initialized)
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/cameras/status', methods=['GET'])
    def camera_status():
        """Obtener estado de las cámaras"""
        try:
            status = {}
            for camera_id in camera_manager.cameras:
                frame = camera_manager.get_frame(camera_id)
                status[camera_id] = {
                    'is_active': frame is not None,
                    'last_frame_received': datetime.now().isoformat() if frame is not None else None
                }
            
            return jsonify({
                'success': True,
                'cameras': status,
                'recording_active': camera_manager.recording_active
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # ENDPOINTS DE GRABACIÓN
    
    @app.route('/api/recording/start', methods=['POST'])
    def start_recording():
        """Iniciar grabación"""
        try:
            data = request.get_json() or {}
            patient_id = data.get('patient_id', '1')
            session_id = data.get('session_id', '1')
            
            # Reiniciar flag de fallo de cámaras al iniciar nueva sesión
            global camera_failure_detected
            camera_failure_detected = False
            
            # Verificar que hay cámaras inicializadas
            if not camera_manager.cameras:
                return jsonify({
                    'success': False,
                    'error': 'No hay cámaras inicializadas. Inicialice las cámaras primero.'
                }), 400
            
            # Iniciar sesión
            result_session_id = video_processor.start_session(patient_id, session_id)
            
            # Notificar al servidor que la sesión inició (el servidor maneja automáticamente el cierre de sesiones anteriores)
            try:
                url = f"{SystemConfig.SERVER.base_url}{SystemConfig.SERVER.session_start_endpoint}"
                start_response = requests.post(url, json={
                    'patient_id': patient_id,
                    'session_id': session_id,  # Usar el session_id del frontend
                    'cameras_count': len(camera_manager.cameras)
                }, timeout=10)
                
                if start_response.status_code != 200:
                    print(f"Error notificando inicio de sesión al servidor: {start_response.status_code}")
            except Exception as e:
                print(f"Error notificando inicio de sesión al servidor: {e}")
                # No fallar si el servidor no responde, continuar con grabación local
            
            # Iniciar grabación
            if video_processor.start_recording():
                return jsonify({
                    'success': True,
                    'session_id': session_id,
                    'patient_id': patient_id,
                    'cameras_recording': list(camera_manager.cameras.keys()),
                    'cameras_initialized': len(camera_manager.cameras),
                    'chunk_duration_seconds': SystemConfig.RECORDING.chunk_duration_seconds
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Error iniciando grabación'
                }), 500
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/recording/status', methods=['GET'])
    def get_recording_status():
        """Obtener estado actual de la grabación"""
        try:
            global camera_failure_detected
            
            # Manejar casos donde video_processor puede no estar inicializado
            try:
                is_recording = video_processor.recording_active if video_processor else False
                session_id = video_processor.session_id if video_processor else None
                patient_id = video_processor.patient_id if video_processor else None
            except AttributeError:
                is_recording = False
                session_id = None
                patient_id = None
            
            cameras_info = []
            try:
                for camera_id, camera in camera_manager.cameras.items():
                    cameras_info.append({
                        'camera_id': camera_id,
                        'is_connected': getattr(camera, 'is_connected', False),
                        'is_recording': getattr(camera, 'is_recording', False)
                    })
            except AttributeError:
                # Si camera_manager.cameras no existe o está vacío
                pass
            
            # Determinar si la sesión fue cancelada por fallo de cámaras
            session_cancelled_by_camera_failure = camera_failure_detected and not is_recording and session_id is None
            
            return jsonify({
                'success': True,
                'is_recording': is_recording,
                'session_id': session_id,
                'patient_id': patient_id,
                'cameras': cameras_info,
                'total_cameras': len(cameras_info),
                'session_cancelled': not is_recording and session_id is None,
                'camera_failure_detected': camera_failure_detected,
                'session_cancelled_by_camera_failure': session_cancelled_by_camera_failure
            })
            
        except Exception as e:
            print(f"Error en get_recording_status: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'camera_failure_detected': camera_failure_detected,
                'session_cancelled_by_camera_failure': camera_failure_detected
            }), 200  # Cambiar a 200 para que el frontend pueda procesar la respuesta

    @app.route('/api/recording/stop', methods=['POST'])
    def stop_recording():
        """Finalizar grabación"""
        try:
            print("Procesando finalización de grabación...")
            final_chunks = video_processor.stop_recording()
            
            # Enviar chunks finales inmediatamente al servidor
            if final_chunks:
                print(f"Enviando {len(final_chunks)} chunks finales al servidor...")
                for chunk in final_chunks:
                    try:
                        # Usar el mismo callback que se usa para chunks regulares
                        upload_chunk_to_server(chunk)
                        print(f"Chunk final enviado: Cámara {chunk.camera_id}, Duración: {chunk.duration_seconds:.2f}s")
                    except Exception as upload_error:
                        print(f"Error enviando chunk final de cámara {chunk.camera_id}: {upload_error}")
            
            # Dar un momento para que se completen las subidas
            time.sleep(2)
            
            # Notificar al servidor que la sesión terminó
            try:
                url = f"{SystemConfig.SERVER.base_url}{SystemConfig.SERVER.session_end_endpoint}"
                end_response = requests.post(url, json={
                    'session_id': video_processor.session_id,
                    'patient_id': video_processor.patient_id,
                    'final_chunks_count': len(final_chunks),
                    'reason': 'session_completed'
                }, timeout=10)
                
                if end_response.status_code == 200:
                    print("Sesión finalizada correctamente en el servidor (datos preservados)")
                elif end_response.status_code == 400:
                    print("Info: No había sesión activa en el servidor para finalizar")
                else:
                    print(f"Warning: Respuesta inesperada del servidor al finalizar: {end_response.status_code}")
            except Exception as e:
                print(f"Error notificando fin de sesión al servidor: {e}")
            
            return jsonify({
                'success': True,
                'session_id': video_processor.session_id,
                'final_chunks_count': len(final_chunks),
                'message': f'Grabación finalizada correctamente. {len(final_chunks)} chunks finales enviados.'
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/recording/cancel', methods=['POST'])
    def cancel_recording():
        """Cancelar grabación"""
        try:
            session_id = video_processor.session_id
            patient_id = video_processor.patient_id
            
            video_processor.cancel_recording()
            
            # Notificar al servidor que la sesión fue cancelada
            try:
                url = f"{SystemConfig.SERVER.base_url}{SystemConfig.SERVER.session_cancel_endpoint}"
                cancel_response = requests.post(url, json={
                    'session_id': session_id,
                    'patient_id': patient_id,
                    'reason': 'cancelled_by_user'
                }, timeout=10)
                
                if cancel_response.status_code == 200:
                    print("Sesión cancelada correctamente en el servidor (datos eliminados)")
                elif cancel_response.status_code == 400:
                    print("Info: No había sesión activa en el servidor para cancelar")
                else:
                    print(f"Warning: Respuesta inesperada del servidor al cancelar: {cancel_response.status_code}")
            except Exception as e:
                print(f"Error notificando cancelación al servidor: {e}")
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'message': 'Grabación cancelada y archivos eliminados'
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/recording/status', methods=['GET'])
    def recording_status():
        """Obtener estado de la grabación"""
        try:
            return jsonify({
                'success': True,
                'recording_active': video_processor.recording_active,
                'session_id': video_processor.session_id,
                'patient_id': video_processor.patient_id,
                'cameras_count': len(camera_manager.cameras)
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # ENDPOINTS DE SISTEMA
    
    @app.route('/api/system/health', methods=['GET'])
    def system_health():
        """Verificar estado del sistema"""
        try:
            return jsonify({
                'success': True,
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'cameras_initialized': len(camera_manager.cameras),
                'recording_active': video_processor.recording_active,
                'temp_dir': SystemConfig.TEMP_VIDEO_DIR,
                'server_config': {
                    'base_url': SystemConfig.SERVER.base_url,
                    'upload_endpoint': SystemConfig.SERVER.upload_endpoint
                }
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'status': 'unhealthy'
            }), 500
    
    @app.route('/api/system/cleanup', methods=['POST'])
    def cleanup_system():
        """Limpiar recursos del sistema"""
        try:
            camera_manager.cleanup()
            
            return jsonify({
                'success': True,
                'message': 'Sistema limpiado correctamente'
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    return app


def run_server(): # Ejecuta el servidor Flask
    
    # Crear directorios necesarios
    SystemConfig.ensure_directories()
    
    app = create_app()
    
    print(f"Iniciando servidor de cámaras Orbbec...")
    print(f"URL: http://{SystemConfig.LOCAL_API_HOST}:{SystemConfig.LOCAL_API_PORT}")
    print(f"Directorio temporal: {SystemConfig.TEMP_VIDEO_DIR}")
    print(f"Servidor de procesamiento: {SystemConfig.SERVER.base_url}")

    app.run(
        host=SystemConfig.LOCAL_API_HOST,
        port=SystemConfig.LOCAL_API_PORT,
        debug=True,
        threaded=True
    )
