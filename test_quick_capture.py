#!/usr/bin/env python3
"""
Script simple para verificar captura básica de color y profundidad
"""

import os
import sys
import cv2
import numpy as np
import time

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.camera_manager.camera_manager import camera_manager
from backend.config.settings import CameraConfig

def quick_test():
    """Prueba rápida de captura"""
    print("🔍 PRUEBA RÁPIDA DE CAPTURA")
    print("-" * 40)
    
    # Descubrir cámaras
    print("1. Descubriendo cámaras...")
    cameras = camera_manager.discover_cameras()
    
    if not cameras:
        print("❌ No hay cámaras disponibles")
        return False
    
    print(f"✅ Encontradas {len(cameras)} cámaras")
    
    # Inicializar cámara
    print("2. Inicializando cámara 0...")
    config = CameraConfig()
    
    if not camera_manager.initialize_camera(0, config):
        print("❌ Error inicializando cámara")
        return False
    
    print("✅ Cámara inicializada")
    
    # Prueba de captura
    print("3. Probando captura (5 intentos)...")
    
    color_ok = 0
    depth_ok = 0
    
    for i in range(5):
        print(f"   Intento {i+1}/5...")
        
        # Color
        color_frame = camera_manager.get_frame(0)
        if color_frame is not None:
            print(f"     ✅ Color: {color_frame.shape}")
            color_ok += 1
            
            # Guardar primera imagen de color
            if i == 0:
                cv2.imwrite("test_color.jpg", color_frame)
                print("     📸 Guardado: test_color.jpg")
        else:
            print("     ❌ Color: FALLO")
        
        # Profundidad
        depth_frame = camera_manager.get_depth_frame(0)
        if depth_frame is not None:
            print(f"     ✅ Depth: {depth_frame.shape}, rango: {np.min(depth_frame)}-{np.max(depth_frame)}")
            depth_ok += 1
            
            # Guardar primera imagen de profundidad
            if i == 0:
                np.save("test_depth.npy", depth_frame)
                print("     💾 Guardado: test_depth.npy")
                
                # Crear visualización
                depth_norm = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
                depth_8bit = np.uint8(depth_norm)
                depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
                cv2.imwrite("test_depth_visual.jpg", depth_colored)
                print("     🎨 Guardado: test_depth_visual.jpg")
        else:
            print("     ❌ Depth: FALLO")
        
        time.sleep(0.5)  # Pausa entre capturas
    
    # Resultados
    print("\n4. Resultados:")
    print(f"   Color exitosos: {color_ok}/5 ({color_ok/5*100:.0f}%)")
    print(f"   Depth exitosos: {depth_ok}/5 ({depth_ok/5*100:.0f}%)")
    
    # Cleanup
    camera_manager.cleanup()
    
    success = color_ok >= 3 and depth_ok >= 3
    
    if success:
        print("\n🎉 PRUEBA EXITOSA: Captura funcionando correctamente")
    else:
        print("\n⚠️  PRUEBA CON PROBLEMAS: Revisar configuración")
    
    return success

if __name__ == "__main__":
    try:
        quick_test()
    except KeyboardInterrupt:
        print("\n⚠️  Interrumpido por usuario")
        camera_manager.cleanup()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        camera_manager.cleanup()
