#!/usr/bin/env python3
"""
Grabación simple con cámaras Orbbec desde cero
Sin usar las clases del proyecto, directamente con pyorbbecsdk
"""
import cv2
import numpy as np
import time
from datetime import datetime
import os

def grabar_video_desde_cero():
    """Grabar video usando directamente pyorbbecsdk"""
    print("="*60)
    print("GRABACIÓN SIMPLE - DESDE CERO CON PYORBBECSDK")
    print("="*60)
    
    try:
        # Importar SDK de Orbbec directamente
        from pyorbbecsdk import Context, Pipeline, Config, OBSensorType, OBFormat
        
        print("✅ SDK Orbbec importado correctamente")
        
        # 1. Inicializar contexto Orbbec
        ctx = Context()
        print("✅ Contexto Orbbec creado")
        
        # 2. Buscar dispositivos (cámaras)
        device_list = ctx.query_devices()
        device_count = device_list.get_count()
        
        if device_count == 0:
            print("❌ No se encontraron cámaras Orbbec")
            return
            
        print(f"✅ Encontradas {device_count} cámaras Orbbec")
        
        # Mostrar información de las cámaras
        for i in range(device_count):
            device = device_list[i]
            serial = device.get_device_info().get_serial_number()
            print(f"   Cámara {i}: S/N {serial}")
        
        # 3. Usar primera cámara
        device = device_list[0]
        pipeline = Pipeline(device)
        print("✅ Pipeline creado")
        
        # 4. Configurar stream de color
        config = Config()
        
        # Obtener perfiles disponibles
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        
        # INTENTAR EVITAR MJPEG - buscar otros formatos primero
        color_profile = None
        
        # Lista de formatos preferidos (evitamos MJPEG)
        formatos_preferidos = [OBFormat.RGB, OBFormat.BGR, OBFormat.YUYV, OBFormat.UYVY]
        
        print("🔍 Buscando formatos compatibles...")
        for formato in formatos_preferidos:
            try:
                # Intentar conseguir un perfil con este formato
                test_profile = profile_list.get_video_stream_profile(640, 480, formato, 30)
                if test_profile:
                    color_profile = test_profile
                    print(f"✅ Formato encontrado: {formato}")
                    break
            except:
                continue
        
        # Si no encontramos ninguno, usar por defecto (probablemente MJPEG)
        if color_profile is None:
            color_profile = profile_list.get_default_video_stream_profile()
            print("⚠️ Usando formato por defecto (posiblemente MJPEG)")
        
        width = color_profile.get_width()
        height = color_profile.get_height()
        fps = color_profile.get_fps()
        formato = color_profile.get_format()
        
        print(f"✅ Perfil seleccionado: {width}x{height}@{fps}fps - {formato}")
        
        # Activar stream
        config.enable_stream(color_profile)
        
        # 5. Iniciar pipeline
        pipeline.start(config)
        print("✅ Pipeline iniciado")
        
        # 6. Configurar grabación de video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"video_orbbec_{timestamp}.mp4"
        
        # Configurar codec y writer de OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            print("❌ Error creando archivo de video")
            pipeline.stop()
            return
            
        print(f"✅ Archivo de video creado: {video_filename}")
        
        # 7. Configurar duración de grabación
        duracion_segundos = 10  # Cambia aquí la duración
        frames_objetivo = fps * duracion_segundos
        frames_grabados = 0
        frames_error = 0
        
        print(f"\n🎬 Iniciando grabación por {duracion_segundos} segundos...")
        print("   Presiona Ctrl+C para detener antes")
        
        tiempo_inicio = time.time()
        
        try:
            while frames_grabados < frames_objetivo:
                # Obtener frameset con timeout
                frameset = pipeline.wait_for_frames(100)  # 100ms timeout
                
                if frameset is None:
                    frames_error += 1
                    continue
                
                # Obtener frame de color
                color_frame = frameset.get_color_frame()
                
                if color_frame is None:
                    frames_error += 1
                    continue
                
                # Convertir frame a array numpy
                frame_data = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
                frame_format = color_frame.get_format()
                
                # Procesar según el formato
                imagen_bgr = None
                
                if frame_format == OBFormat.RGB:
                    # RGB -> reshape y convertir a BGR
                    imagen = frame_data.reshape((height, width, 3))
                    imagen_bgr = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
                    
                elif frame_format == OBFormat.BGR:
                    # BGR directo
                    imagen_bgr = frame_data.reshape((height, width, 3))
                    
                elif frame_format == OBFormat.MJPG:
                    # MJPEG -> decodificar con diagnósticos mejorados
                    if frames_grabados == 0:  # Solo para el primer frame
                        print(f"   🔍 Debug MJPEG: data_size={len(frame_data)}, min={frame_data.min()}, max={frame_data.max()}")
                        print(f"   🔍 Primeros 20 bytes: {frame_data[:20]}")
                    
                    # Verificar si los datos son válidos
                    if len(frame_data) < 100:  # MJPEG muy pequeño
                        if frames_grabados == 0:
                            print(f"   ⚠️ Frame MJPEG muy pequeño: {len(frame_data)} bytes")
                        frames_error += 1
                        continue
                    
                    # Verificar cabeceras JPEG típicas
                    if not (frame_data[0] == 0xFF and frame_data[1] == 0xD8):
                        if frames_grabados == 0:
                            print(f"   ⚠️ Sin cabecera JPEG válida: {hex(frame_data[0])}{hex(frame_data[1])}")
                        frames_error += 1
                        continue
                    
                    # Intentar decodificar
                    imagen_bgr = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
                    if imagen_bgr is None:
                        frames_error += 1
                        continue
                        
                elif frame_format == OBFormat.YUYV:
                    # YUYV -> convertir a BGR
                    try:
                        yuyv_data = frame_data.reshape((height, width * 2))
                        imagen_bgr = cv2.cvtColor(yuyv_data, cv2.COLOR_YUV2BGR_YUYV)
                    except Exception as e:
                        if frames_grabados == 0:
                            print(f"   ❌ Error YUYV: {e}")
                        frames_error += 1
                        continue
                    
                else:
                    if frames_grabados == 0:  # Solo mostrar una vez
                        print(f"⚠️ Formato no soportado: {frame_format}")
                    frames_error += 1
                    continue
                
                # Verificar que la imagen sea válida
                if imagen_bgr is None or imagen_bgr.size == 0:
                    frames_error += 1
                    continue
                
                # Verificar dimensiones
                if imagen_bgr.shape[:2] != (height, width):
                    if frames_grabados == 0:
                        print(f"   ⚠️ Dimensiones incorrectas: esperado {height}x{width}, obtenido {imagen_bgr.shape}")
                    frames_error += 1
                    continue
                
                # Escribir frame al video
                video_writer.write(imagen_bgr)
                frames_grabados += 1
                
                # Mostrar progreso
                if frames_grabados % fps == 0:
                    segundos_transcurridos = frames_grabados // fps
                    print(f"   📹 {segundos_transcurridos}/{duracion_segundos} segundos - Frames: {frames_grabados}")
                
        except KeyboardInterrupt:
            print("\n⏹️ Grabación detenida por usuario")
            
        # 8. Finalizar y limpiar
        tiempo_final = time.time()
        tiempo_total = tiempo_final - tiempo_inicio
        
        video_writer.release()
        pipeline.stop()
        
        print(f"\n✅ GRABACIÓN COMPLETADA")
        print(f"   📁 Archivo: {video_filename}")
        print(f"   🎞️ Frames grabados: {frames_grabados}")
        print(f"   ❌ Frames con error: {frames_error}")
        print(f"   ⏱️ Tiempo total: {tiempo_total:.1f} segundos")
        
        if frames_grabados > 0:
            print(f"   📊 FPS promedio: {frames_grabados/tiempo_total:.1f}")
        
        # Verificar archivo
        if os.path.exists(video_filename):
            size_mb = os.path.getsize(video_filename) / (1024 * 1024)
            print(f"   💾 Tamaño: {size_mb:.2f} MB")
            
            if size_mb < 0.1:
                print("   ⚠️ Archivo muy pequeño, posible problema en grabación")
        else:
            print("   ❌ Error: Archivo no creado")
        
    except ImportError as e:
        print(f"❌ Error importando SDK: {e}")
        print("   Instala pyorbbecsdk correctamente")
        
    except Exception as e:
        print(f"❌ Error durante grabación: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    grabar_video_desde_cero()
