#!/usr/bin/env python3
"""
Script de verificación de integridad de captura
Verifica que el color y la profundidad se capturen correctamente
"""

import os
import sys
import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

# Agregar el directorio raíz al path para importar módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.camera_manager.camera_manager import camera_manager
from backend.config.settings import CameraConfig

class CaptureIntegrityTest:
    def __init__(self):
        self.test_dir = Path("test_capture_results")
        self.test_dir.mkdir(exist_ok=True)
        
    def run_test(self, duration_seconds=10):
        """Ejecutar prueba de integridad de captura"""
        print("=" * 60)
        print("PRUEBA DE INTEGRIDAD DE CAPTURA - COLOR Y PROFUNDIDAD")
        print("=" * 60)
        
        # Paso 1: Descubrir cámaras
        print("\n1. Descubriendo cámaras...")
        cameras = camera_manager.discover_cameras()
        
        if not cameras:
            print("❌ No se encontraron cámaras")
            return False
            
        print(f"✅ Encontradas {len(cameras)} cámaras:")
        for cam in cameras:
            print(f"   - Cámara {cam.camera_id}: S/N {cam.serial_number}")
        
        # Paso 2: Inicializar primera cámara
        print("\n2. Inicializando cámara 0...")
        config = CameraConfig()
        
        if not camera_manager.initialize_camera(0, config):
            print("❌ Error inicializando cámara 0")
            return False
            
        print("✅ Cámara 0 inicializada correctamente")
        
        # Paso 3: Prueba de captura de frames individuales
        print("\n3. Probando captura de frames individuales...")
        
        color_frames = []
        depth_frames = []
        timestamps = []
        
        print(f"Capturando frames durante {duration_seconds} segundos...")
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration_seconds:
            # Capturar frame de color
            color_frame = camera_manager.get_frame(0)
            
            # Capturar frame de profundidad
            depth_frame = camera_manager.get_depth_frame(0)
            
            current_time = time.time() - start_time
            
            if color_frame is not None:
                color_frames.append(color_frame.copy())
                print(f"✅ Frame color {frame_count}: {color_frame.shape} - {current_time:.2f}s")
            else:
                print(f"❌ Frame color {frame_count}: FALLO - {current_time:.2f}s")
            
            if depth_frame is not None:
                depth_frames.append(depth_frame.copy())
                print(f"✅ Frame depth {frame_count}: {depth_frame.shape} - {current_time:.2f}s")
            else:
                print(f"❌ Frame depth {frame_count}: FALLO - {current_time:.2f}s")
            
            timestamps.append(current_time)
            frame_count += 1
            
            # Esperar para simular ~30 FPS
            time.sleep(1/30)
        
        # Paso 4: Análisis de resultados
        print(f"\n4. Análisis de resultados:")
        print(f"   - Total frames intentados: {frame_count}")
        print(f"   - Frames color capturados: {len(color_frames)}")
        print(f"   - Frames depth capturados: {len(depth_frames)}")
        print(f"   - Tasa éxito color: {len(color_frames)/frame_count*100:.1f}%")
        print(f"   - Tasa éxito depth: {len(depth_frames)/frame_count*100:.1f}%")
        
        # Paso 5: Guardar muestras para inspección visual
        print("\n5. Guardando muestras para inspección...")
        
        if color_frames:
            self.save_color_samples(color_frames[:5])  # Primeros 5 frames
        
        if depth_frames:
            self.save_depth_samples(depth_frames[:5])  # Primeros 5 frames
        
        # Paso 6: Verificar calidad de datos
        print("\n6. Verificando calidad de datos...")
        
        color_quality = self.analyze_color_quality(color_frames)
        depth_quality = self.analyze_depth_quality(depth_frames)
        
        print(f"   - Calidad color: {'✅ BUENA' if color_quality else '❌ PROBLEMAS'}")
        print(f"   - Calidad depth: {'✅ BUENA' if depth_quality else '❌ PROBLEMAS'}")
        
        # Paso 7: Crear gráfico de estadísticas
        self.create_statistics_plot(color_frames, depth_frames, timestamps)
        
        # Limpieza
        camera_manager.cleanup()
        
        # Resultado final
        success = (len(color_frames) > frame_count * 0.8 and 
                  len(depth_frames) > frame_count * 0.8 and
                  color_quality and depth_quality)
        
        print(f"\n{'='*60}")
        if success:
            print("🎉 PRUEBA EXITOSA: Captura de color y profundidad funcionando correctamente")
        else:
            print("⚠️  PRUEBA CON PROBLEMAS: Revisar configuración de cámara")
        print(f"{'='*60}")
        
        return success
    
    def save_color_samples(self, frames):
        """Guardar muestras de frames de color"""
        print("   Guardando muestras de color...")
        
        for i, frame in enumerate(frames):
            if frame is not None:
                # Guardar como imagen
                filename = self.test_dir / f"color_sample_{i+1}.jpg"
                cv2.imwrite(str(filename), frame)
                
                # Información del frame
                height, width = frame.shape[:2]
                print(f"     - Muestra {i+1}: {width}x{height} - {filename}")
    
    def save_depth_samples(self, frames):
        """Guardar muestras de frames de profundidad"""
        print("   Guardando muestras de profundidad...")
        
        for i, frame in enumerate(frames):
            if frame is not None:
                # Guardar como numpy array
                npy_filename = self.test_dir / f"depth_sample_{i+1}.npy"
                np.save(str(npy_filename), frame)
                
                # Crear visualización de profundidad
                img_filename = self.test_dir / f"depth_sample_{i+1}.png"
                self.create_depth_visualization(frame, img_filename)
                
                # Información del frame
                height, width = frame.shape[:2]
                min_val, max_val = np.min(frame), np.max(frame)
                print(f"     - Muestra {i+1}: {width}x{height}, rango: {min_val}-{max_val} - {npy_filename}")
    
    def create_depth_visualization(self, depth_frame, filename):
        """Crear visualización colorizada del frame de profundidad"""
        try:
            # Normalizar datos de profundidad
            depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
            depth_8bit = np.uint8(depth_normalized)
            
            # Aplicar mapa de colores
            depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
            
            # Guardar imagen colorizada
            cv2.imwrite(str(filename), depth_colored)
        except Exception as e:
            print(f"     Error creando visualización de profundidad: {e}")
    
    def analyze_color_quality(self, frames):
        """Analizar calidad de frames de color"""
        if not frames:
            return False
        
        try:
            # Verificar que no sean frames negros
            for frame in frames[:3]:  # Revisar primeros 3 frames
                mean_intensity = np.mean(frame)
                if mean_intensity < 10:  # Frame muy oscuro
                    print(f"     ⚠️  Frame color muy oscuro (intensidad: {mean_intensity:.1f})")
                    return False
            
            # Verificar dimensiones consistentes
            first_shape = frames[0].shape
            for frame in frames:
                if frame.shape != first_shape:
                    print(f"     ⚠️  Dimensiones inconsistentes: {frame.shape} vs {first_shape}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"     Error analizando calidad color: {e}")
            return False
    
    def analyze_depth_quality(self, frames):
        """Analizar calidad de frames de profundidad"""
        if not frames:
            return False
        
        try:
            # Verificar que no sean frames vacíos
            for frame in frames[:3]:  # Revisar primeros 3 frames
                non_zero_pixels = np.count_nonzero(frame)
                total_pixels = frame.size
                
                if non_zero_pixels < total_pixels * 0.1:  # Menos del 10% de datos válidos
                    print(f"     ⚠️  Frame depth con pocos datos válidos: {non_zero_pixels}/{total_pixels}")
                    return False
            
            # Verificar dimensiones consistentes
            first_shape = frames[0].shape
            for frame in frames:
                if frame.shape != first_shape:
                    print(f"     ⚠️  Dimensiones depth inconsistentes: {frame.shape} vs {first_shape}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"     Error analizando calidad depth: {e}")
            return False
    
    def create_statistics_plot(self, color_frames, depth_frames, timestamps):
        """Crear gráfico de estadísticas de captura"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Gráfico 1: Frames capturados vs tiempo
            color_times = timestamps[:len(color_frames)]
            depth_times = timestamps[:len(depth_frames)]
            
            ax1.plot(color_times, range(len(color_frames)), 'b-', label='Color', linewidth=2)
            ax1.plot(depth_times, range(len(depth_frames)), 'r-', label='Depth', linewidth=2)
            ax1.set_xlabel('Tiempo (s)')
            ax1.set_ylabel('Frames capturados')
            ax1.set_title('Frames Capturados vs Tiempo')
            ax1.legend()
            ax1.grid(True)
            
            # Gráfico 2: Histograma de intensidades de color
            if color_frames:
                intensities = [np.mean(frame) for frame in color_frames[:10]]
                ax2.hist(intensities, bins=20, alpha=0.7, color='blue')
                ax2.set_xlabel('Intensidad media')
                ax2.set_ylabel('Frecuencia')
                ax2.set_title('Distribución de Intensidades (Color)')
                ax2.grid(True)
            
            # Gráfico 3: Estadísticas de profundidad
            if depth_frames:
                depth_means = [np.mean(frame[frame > 0]) for frame in depth_frames[:10] if np.any(frame > 0)]
                if depth_means:
                    ax3.plot(depth_means, 'ro-', linewidth=2, markersize=6)
                    ax3.set_xlabel('Frame #')
                    ax3.set_ylabel('Profundidad media')
                    ax3.set_title('Profundidad Media por Frame')
                    ax3.grid(True)
            
            # Gráfico 4: Resumen de tasas de éxito
            total_attempted = len(timestamps)
            success_rates = [
                ('Color', len(color_frames) / total_attempted * 100),
                ('Depth', len(depth_frames) / total_attempted * 100)
            ]
            
            labels, rates = zip(*success_rates)
            colors = ['blue', 'red']
            bars = ax4.bar(labels, rates, color=colors, alpha=0.7)
            ax4.set_ylabel('Tasa de éxito (%)')
            ax4.set_title('Tasas de Éxito de Captura')
            ax4.set_ylim(0, 100)
            
            # Añadir valores en las barras
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Guardar gráfico
            plot_filename = self.test_dir / "capture_statistics.png"
            plt.savefig(str(plot_filename), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   📊 Gráfico de estadísticas guardado: {plot_filename}")
            
        except Exception as e:
            print(f"   Error creando gráfico de estadísticas: {e}")

def main():
    """Función principal"""
    print("Iniciando prueba de integridad de captura...")
    
    try:
        # Crear instancia del test
        test = CaptureIntegrityTest()
        
        # Ejecutar prueba (10 segundos por defecto)
        duration = 10
        if len(sys.argv) > 1:
            try:
                duration = int(sys.argv[1])
            except ValueError:
                print("Duración inválida, usando 10 segundos por defecto")
        
        print(f"Duración de la prueba: {duration} segundos")
        
        # Ejecutar test
        success = test.run_test(duration)
        
        # Mostrar ubicación de resultados
        print(f"\n📁 Resultados guardados en: {test.test_dir.absolute()}")
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Prueba interrumpida por el usuario")
        camera_manager.cleanup()
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error durante la prueba: {e}")
        import traceback
        traceback.print_exc()
        camera_manager.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()
