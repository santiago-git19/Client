#!/usr/bin/env python3
"""
Visualizador en tiempo real de frames de color y profundidad con colormap
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

def visualize_depth_live():
    """Visualización en vivo de color y profundidad"""
    print("🎥 VISUALIZADOR EN TIEMPO REAL")
    print("-" * 40)
    print("Controles:")
    print("  - ESC o Q: Salir")
    print("  - S: Guardar frame actual")
    print("  - C: Cambiar colormap de profundidad")
    print("  - R: Resetear rango de profundidad")
    print()
    
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
    print("3. Iniciando visualización...")
    
    # Variables para visualización
    colormaps = [
        cv2.COLORMAP_JET,
        cv2.COLORMAP_VIRIDIS,
        cv2.COLORMAP_PLASMA,
        cv2.COLORMAP_INFERNO,
        cv2.COLORMAP_MAGMA,
        cv2.COLORMAP_HOT,
        cv2.COLORMAP_COOL,
        cv2.COLORMAP_RAINBOW
    ]
    colormap_names = [
        "JET", "VIRIDIS", "PLASMA", "INFERNO", 
        "MAGMA", "HOT", "COOL", "RAINBOW"
    ]
    current_colormap = 0
    
    # Variables para normalización automática
    min_depth = None
    max_depth = None
    auto_range = True
    
    frame_count = 0
    save_count = 0
    start_time = time.time()
    
    try:
        while True:
            frame_count += 1
            
            # Capturar frames
            color_frame = camera_manager.get_frame(0)
            depth_frame = camera_manager.get_depth_frame(0)
            
            # Crear ventanas de visualización
            display_color = None
            display_depth = None
            info_text = []
            
            # Procesar frame de color
            if color_frame is not None:
                display_color = color_frame.copy()
                
                # Añadir información
                cv2.putText(display_color, f"COLOR - Frame: {frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_color, f"Shape: {color_frame.shape}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                info_text.append(f"✅ Color: {color_frame.shape}")
            else:
                # Frame negro si no hay color
                display_color = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(display_color, "NO COLOR FRAME", 
                           (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                info_text.append("❌ Color: FALLO")
            
            # Procesar frame de profundidad
            if depth_frame is not None:
                # Estadísticas del frame de profundidad
                valid_mask = depth_frame > 0
                valid_depths = depth_frame[valid_mask]
                
                if len(valid_depths) > 0:
                    frame_min = np.min(valid_depths)
                    frame_max = np.max(valid_depths)
                    frame_mean = np.mean(valid_depths)
                    
                    # Actualizar rango automático
                    if auto_range:
                        if min_depth is None or frame_min < min_depth:
                            min_depth = frame_min
                        if max_depth is None or frame_max > max_depth:
                            max_depth = frame_max
                    
                    # Normalizar profundidad
                    if min_depth is not None and max_depth is not None and max_depth > min_depth:
                        # Usar rango automático o manual
                        norm_min = min_depth
                        norm_max = max_depth
                    else:
                        # Usar rango del frame actual
                        norm_min = frame_min
                        norm_max = frame_max
                    
                    # Crear máscara para valores válidos
                    depth_normalized = np.zeros_like(depth_frame, dtype=np.float32)
                    depth_normalized[valid_mask] = (depth_frame[valid_mask] - norm_min) / (norm_max - norm_min)
                    depth_normalized = np.clip(depth_normalized, 0, 1)
                    
                    # Convertir a 8-bit
                    depth_8bit = (depth_normalized * 255).astype(np.uint8)
                    
                    # Aplicar colormap
                    display_depth = cv2.applyColorMap(depth_8bit, colormaps[current_colormap])
                    
                    # Marcar píxeles inválidos en negro
                    display_depth[~valid_mask] = [0, 0, 0]
                    
                    # Añadir información
                    cv2.putText(display_depth, f"DEPTH - {colormap_names[current_colormap]}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_depth, f"Range: {norm_min:.0f}-{norm_max:.0f}mm", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(display_depth, f"Frame: {frame_min:.0f}-{frame_max:.0f}mm", 
                               (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(display_depth, f"Mean: {frame_mean:.0f}mm", 
                               (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(display_depth, f"Valid: {len(valid_depths)}/{depth_frame.size}", 
                               (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    info_text.append(f"✅ Depth: {depth_frame.shape}, válidos: {len(valid_depths)}")
                    info_text.append(f"   Rango: {frame_min:.0f}-{frame_max:.0f}mm, Media: {frame_mean:.0f}mm")
                
                else:
                    # Frame de profundidad sin datos válidos
                    display_depth = np.zeros((depth_frame.shape[0], depth_frame.shape[1], 3), dtype=np.uint8)
                    cv2.putText(display_depth, "NO VALID DEPTH DATA", 
                               (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    info_text.append("⚠️  Depth: Sin datos válidos")
                    
            else:
                # Frame negro si no hay profundidad
                display_depth = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(display_depth, "NO DEPTH FRAME", 
                           (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                info_text.append("❌ Depth: FALLO")
            
            # Mostrar frames
            cv2.imshow("Color Frame", display_color)
            cv2.imshow("Depth Frame (Colored)", display_depth)
            
            # Información en consola cada 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"\n📊 Frame {frame_count} - FPS: {fps:.1f}")
                for line in info_text:
                    print(f"   {line}")
                print(f"   Colormap: {colormap_names[current_colormap]}")
                if min_depth is not None and max_depth is not None:
                    print(f"   Rango acumulado: {min_depth:.0f}-{max_depth:.0f}mm")
            
            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27 or key == ord('q'):  # ESC o Q
                break
            elif key == ord('s'):  # Guardar
                if display_color is not None:
                    cv2.imwrite(f"live_color_{save_count:03d}.jpg", color_frame)
                if display_depth is not None:
                    cv2.imwrite(f"live_depth_colored_{save_count:03d}.jpg", display_depth)
                    if depth_frame is not None:
                        np.save(f"live_depth_raw_{save_count:03d}.npy", depth_frame)
                print(f"💾 Guardado frame {save_count}")
                save_count += 1
            elif key == ord('c'):  # Cambiar colormap
                current_colormap = (current_colormap + 1) % len(colormaps)
                print(f"🎨 Colormap: {colormap_names[current_colormap]}")
            elif key == ord('r'):  # Reset rango
                min_depth = None
                max_depth = None
                print("🔄 Rango de profundidad reseteado")
            
            time.sleep(0.033)  # ~30 FPS
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrumpido por usuario")
    
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        camera_manager.cleanup()
        
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\n📈 Estadísticas finales:")
        print(f"   Frames procesados: {frame_count}")
        print(f"   Tiempo total: {elapsed:.1f}s")
        print(f"   FPS promedio: {fps:.1f}")
        print(f"   Frames guardados: {save_count}")
    
    return True

if __name__ == "__main__":
    try:
        visualize_depth_live()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
        camera_manager.cleanup()
