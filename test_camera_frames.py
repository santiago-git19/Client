#!/usr/bin/env python3
"""
Test simple para verificar captura de frames desde la cámara Orbbec
"""

import os
import sys
import cv2
import time

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.camera_manager import camera_manager
from backend.config.settings import CameraConfig

def test_camera_frames():
    """Test simple de captura de frames"""
    print("🔍 TEST DE CAPTURA DE FRAMES")
    print("-" * 40)
    
    try:
        # 1. Descubrir cámaras
        print("1. Descubriendo cámaras...")
        cameras = camera_manager.discover_cameras()
        
        if not cameras:
            print("❌ No hay cámaras disponibles")
            return False
        
        print(f"✅ Encontradas {len(cameras)} cámaras")
        
        # 2. Inicializar cámara 0
        print("2. Inicializando cámara 0...")
        config = CameraConfig()
        
        if not camera_manager.initialize_camera(0, config):
            print("❌ Error inicializando cámara")
            return False
        
        print("✅ Cámara inicializada")
        
        # 3. Test básico de frames SIN activar grabación
        print("3. Test de frames sin modo grabación...")
        success_count = 0
        
        for i in range(10):
            print(f"   Intento {i+1}/10...")
            frame = camera_manager.get_frame(0)
            
            if frame is not None:
                print(f"     ✅ Frame obtenido: {frame.shape}")
                success_count += 1
            else:
                print("     ❌ Frame falló")
            
            time.sleep(0.1)
        
        print(f"\nResultado SIN grabación: {success_count}/10 frames exitosos")
        
        # 4. Test con modo grabación activado
        print("\n4. Test de frames CON modo grabación...")
        if not camera_manager.start_recording_all():
            print("❌ Error activando grabación")
            return False
        
        print("✅ Modo grabación activado")
        
        success_count_recording = 0
        
        for i in range(10):
            print(f"   Intento {i+1}/10...")
            frame = camera_manager.get_frame(0)
            
            if frame is not None:
                print(f"     ✅ Frame obtenido: {frame.shape}")
                success_count_recording += 1
            else:
                print("     ❌ Frame falló")
            
            time.sleep(0.1)
        
        print(f"\nResultado CON grabación: {success_count_recording}/10 frames exitosos")
        
        # 5. Test de profundidad
        print("\n5. Test de frames de profundidad...")
        depth_success = 0
        
        for i in range(5):
            print(f"   Intento depth {i+1}/5...")
            depth_frame = camera_manager.get_depth_frame(0)
            
            if depth_frame is not None:
                print(f"     ✅ Depth obtenido: {depth_frame.shape}")
                depth_success += 1
            else:
                print("     ❌ Depth falló")
            
            time.sleep(0.1)
        
        print(f"\nResultado PROFUNDIDAD: {depth_success}/5 frames exitosos")
        
        # 6. Cleanup
        camera_manager.stop_recording_all()
        print("\n6. Cleanup completado")
        
        # 7. Análisis de resultados
        print("\n" + "="*50)
        print("📊 ANÁLISIS DE RESULTADOS:")
        print(f"   Sin grabación: {success_count}/10 ({success_count/10*100:.0f}%)")
        print(f"   Con grabación: {success_count_recording}/10 ({success_count_recording/10*100:.0f}%)")
        print(f"   Profundidad: {depth_success}/5 ({depth_success/5*100:.0f}%)")
        
        if success_count_recording == 0:
            print("\n⚠️  PROBLEMA DETECTADO:")
            print("   La cámara NO proporciona frames en modo grabación")
            print("   Posibles causas:")
            print("   - Pipeline no iniciado correctamente")
            print("   - Conflicto de streaming")
            print("   - Configuración de perfil incorrecta")
            return False
        elif success_count_recording < success_count:
            print("\n⚠️  DEGRADACIÓN DETECTADA:")
            print("   La cámara funciona PEOR en modo grabación")
            return False
        else:
            print("\n🎉 CÁMARA FUNCIONANDO CORRECTAMENTE")
            return True
            
    except Exception as e:
        print(f"\n❌ Error en test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        test_camera_frames()
    except KeyboardInterrupt:
        print("\n⚠️  Interrumpido por usuario")
        camera_manager.cleanup()
    except Exception as e:
        print(f"❌ Error: {e}")
        camera_manager.cleanup()
