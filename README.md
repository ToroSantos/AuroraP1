# 📖 Aurora V7 - Sistema Modular de Audio Neuroacústico

Aurora V7 es un sistema modular de audio neuroacústico que produce experiencias sonoras terapéuticas. Su estructura modular y flexible permite crear experiencias de audio personalizadas y de alta calidad.

---

## 🎯 Objetivo del Sistema Aurora

El objetivo principal de Aurora V7 es generar audio neuroacústico que estimule estados mentales y emocionales específicos (relajación, enfoque, introspección, creatividad, etc.) mediante el uso de ondas binaurales, modulación AM/FM, pads emocionales y texturas armónicas. Todo esto está diseñado para ser modular, expansible y adaptable a diferentes contextos terapéuticos y creativos.

Aurora permite:
✅ Modularidad en los motores de audio.  
✅ Integración con sistemas de análisis y calidad.  
✅ Flexibilidad en configuraciones emocionales y estructurales.  
✅ Control programático a través de API y consola.

---

## 🧩 Funcionamiento del Sistema Aurora

Aurora V7 funciona a través de un flujo orquestado que integra diferentes componentes y motores. El flujo general es el siguiente:

```
AuroraDirector (Orquestador principal)
    ↓
Motores Puros: NeuroMix + HyperMod + HarmonicEssence
    ↓
emotion_style_profiles (Configuraciones y Perfiles Emocionales)
    ↓
field_profiles (Presets de Campo)
    ↓
presets_fases (Fases Estructuradas de Audio)
    ↓
Audio Final (Exportación WAV)
```

1. **AuroraDirector (Orquestador Principal)**:  
   Coordina la ejecución de los motores, valida sus instancias y aplica parches si es necesario.

2. **Motores Puros**:  
   - **NeuroMix**: Genera el audio neuroacústico (ondas binaurales, neurotransmisores).  
   - **HyperMod**: Controla la estructura de las fases (progresión emocional y dinámica).  
   - **HarmonicEssence**: Crea las texturas emocionales y estéticas.

3. **Módulos Complementarios**:  
   - **emotion_style_profiles**: Define los perfiles emocionales.  
   - **field_profiles**: Define presets de campo y ajustes base.  
   - **presets_fases**: Organiza el audio en fases estructuradas.

4. **Análisis y Calidad**:  
   - **Carmine_Analyzer** y **aurora_quality_pipeline** analizan la calidad técnica y emocional.  
   - **verify_structure.py** valida la estructura científica y la coherencia global del audio.

5. **Exportación**:  
   El audio final se exporta como archivo WAV o flujo de audio listo para usar.

---

## ✅ Archivos para GUI
- **aurora_flask_app.py**: Implementa la API web (API HTTP).  
- **aurora_bridge.py**: Puente de integración (Conector entre la API y el núcleo de Aurora).

---

## ✅ AuroraDirector (Orquestador principal)
- **aurora_director_v7.py**: Orquesta objetivos, fases y motores.  
- **aurora_multimotor_extension.py**: Gestión inteligente de motores.

---

## ✅ Motores principales
- **neuromix_aurora_v27.py**: Motor neuroacústico.  
- **hypermod_v32.py**: Estructura de fases.  
- **harmonicEssence_v34.py**: Texturas emocionales y estéticas.  
- **neuromix_definitivo_v7.py**: Parche de optimización del motor NeuroMix.  
- **harmonic_essence_optimizations.py**: Parche de optimización del motor HarmonicEssence.

---

## ✅ ANÁLISIS Y CALIDAD
- **Carmine_Analyzer.py**: Análisis emocional y estructural.  
- **aurora_quality_pipeline.py**: Normalización, compresión y mastering.  
- **verify_structure.py**: Validación científica con benchmark y reportes detallados.  
- **carmine_power_booster_v7.py**: Parche de optimización del motor Carmine.

---

## ✅ Módulos de soporte
- **objective_manager.py**: Gestión de objetivos.  
- **emotion_style_profiles.py**: Presets emocionales.  
- **sync_and_scheduler.py**: Sincronización avanzada.  
- **field_profiles.py**: Presets y configuraciones base.  
- **presets_fases.py**: Fases estructuradas.  
- **harmony_generator.py**: Generación de pads armónicos.  
- **psychedelic_effects_tables.json**: Tablas informativas.

---

## ✅ Documentación
- **README.md**: Este archivo con toda la explicación.

---

Aurora V7 es la solución ideal para crear experiencias de audio terapéuticas y de alta calidad de forma modular y escalable.
