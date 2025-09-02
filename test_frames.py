#!/usr/bin/env python3
"""
Script de prueba para verificar la conversión de frames Orbbec
"""
import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Agregar rutas
client_path = Path(__file__).parent
sys.path.insert(0, str(client_path / "backend"))

def test_camera_frames():
    """Probar captura y conversión de frames"""
    print("="*50)
    print("PRUEBA DE CONVERSIÓN DE FRAMES ORBBEC")
    print("="*50)
    
    try:
        from camera_manager.camera_manager import CameraManager
        
        print("✅ CameraManager importado correctamente")
        
        # Descubrir cámaras
        available_cameras = CameraManager.discover_cameras()
        print(f"Cámaras encontradas: {len(available_cameras)}")
        
        if not available_cameras:
            print("❌ No se encontraron cámaras")
            return
        
        # Usar primera cámara
        camera_id = 0
        camera = CameraManager(camera_id)
        
        print(f"Inicializando cámara {camera_id}...")
        if not camera.initialize():
            print(f"❌ No se pudo inicializar cámara {camera_id}")
            return
            
        print(f"✅ Cámara {camera_id} inicializada")
        
        # Capturar algunos frames
        print("Capturando frames de prueba...")
        successful_captures = 0
        
        for i in range(10):
            print(f"\n--- Frame {i+1} ---")
            frame = camera.get_frame()
            
            if frame is not None:
                print(f"✅ Frame capturado: {frame.shape}, dtype={frame.dtype}")
                print(f"   Min: {frame.min()}, Max: {frame.max()}, Mean: {frame.mean():.2f}")
                
                # Verificar si no es gris (valores muy bajos)
                if frame.mean() < 10:
                    print(f"⚠️  Frame muy oscuro (posible problema de conversión)")
                else:
                    print(f"✅ Frame parece válido")
                
                # Guardar primer frame válido para inspección
                if successful_captures == 0:
                    cv2.imwrite("test_frame.jpg", frame)
                    print(f"   Frame guardado como test_frame.jpg")
                
                successful_captures += 1
            else:
                print(f"❌ Frame {i+1} falló")
                
        print(f"\n--- Resumen ---")
        print(f"Frames exitosos: {successful_captures}/10")
        
        if successful_captures > 5:
            print("✅ Conversión funcionando correctamente")
        elif successful_captures > 0:
            print("⚠️  Conversión parcialmente funcional")
        else:
            print("❌ Conversión fallando completamente")
        
        # Cleanup
        camera.cleanup()
        
    except Exception as e:
        print(f"❌ Error durante la prueba: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_camera_frames()
