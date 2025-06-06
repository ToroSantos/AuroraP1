import numpy as np,warnings,time;from scipy import signal as ss;from scipy.stats import kurtosis,skew;from typing import Dict,List,Tuple,Optional,Callable,Any;from dataclasses import dataclass,field;from functools import lru_cache;import logging
logger=logging.getLogger("AuroraQualityPipeline")
@dataclass
class ResultadoFase:
 audio:np.ndarray;metadata:Dict[str,Any];dynamic_range:Optional[float]=None;spectral_balance:Optional[Dict[str,float]]=None;phase_coherence:Optional[float]=None;processing_time:Optional[float]=None
@dataclass
class AnalysisConfig:
 enable_spectral_analysis:bool=True;enable_dynamic_analysis:bool=True;enable_phase_analysis:bool=True;quality_threshold:float=85.0;fft_size:int=4096;overlap:float=0.5;enable_artifact_detection:bool=True;enable_noise_analysis:bool=True;enable_advanced_lufs:bool=True
class ModoProcesamiento:
 PROFESIONAL="PROFESIONAL";TEST="TEST";DEBUG="DEBUG";MASTERING="MASTERING";REALTIME="REALTIME";BATCH="BATCH"
class CarmineAnalyzerBasic:
 def __init__(self,config:Optional[AnalysisConfig]=None):self.config=config or AnalysisConfig();self._analysis_cache={};self._fft_cache={};self._window_cache={}
 @staticmethod
 def evaluar(audio:np.ndarray,sample_rate:int=44100)->Dict[str,Any]:return CarmineAnalyzerBasic().evaluar_avanzado(audio,sample_rate)
 def evaluar_avanzado(self,audio:np.ndarray,sample_rate:int=44100)->Dict[str,Any]:
  start_time=time.time()
  try:
   if not self._validate_audio(audio):return self._get_error_result("Audio inválido")
   resultado={"score":0,"nivel":"desconocido","problemas":[],"recomendaciones":[],"metricas":{}}
   dynamic_score,dynamic_issues=self._analyze_dynamics_enhanced(audio,sample_rate);resultado["metricas"]["dynamic_range"]=dynamic_score;resultado["problemas"].extend(dynamic_issues)
   if self.config.enable_spectral_analysis:spectral_score,spectral_issues=self._analyze_spectrum_enhanced(audio,sample_rate);resultado["metricas"]["spectral_balance"]=spectral_score;resultado["problemas"].extend(spectral_issues)
   if audio.ndim>1 and self.config.enable_phase_analysis:phase_score,phase_issues=self._analyze_phase_coherence_enhanced(audio);resultado["metricas"]["phase_coherence"]=phase_score;resultado["problemas"].extend(phase_issues)
   distortion_score,distortion_issues=self._analyze_distortion_enhanced(audio,sample_rate);resultado["metricas"]["thd"]=distortion_score;resultado["problemas"].extend(distortion_issues)
   if self.config.enable_advanced_lufs:lufs_score,lufs_issues=self._analyze_lufs_precise(audio,sample_rate);resultado["metricas"]["lufs_analysis"]=lufs_score;resultado["problemas"].extend(lufs_issues)
   if self.config.enable_artifact_detection:artifact_score,artifact_issues=self._detect_digital_artifacts(audio,sample_rate);resultado["metricas"]["artifacts"]=artifact_score;resultado["problemas"].extend(artifact_issues)
   if self.config.enable_noise_analysis:noise_score,noise_issues=self._analyze_noise_floor(audio,sample_rate);resultado["metricas"]["noise_floor"]=noise_score;resultado["problemas"].extend(noise_issues)
   resultado["score"]=self._calculate_overall_score_enhanced(resultado["metricas"]);resultado["nivel"]=self._get_quality_level(resultado["score"]);resultado["recomendaciones"]=self._generate_recommendations_enhanced(resultado);resultado["processing_time"]=time.time()-start_time;return resultado
  except Exception as e:logger.error(f"Error en análisis: {e}");return self._get_error_result(str(e))
 def _validate_audio(self,audio:np.ndarray)->bool:
  if not isinstance(audio,np.ndarray):return False
  if audio.size==0:return False
  if np.all(audio==0):return False
  if np.any(np.isnan(audio))or np.any(np.isinf(audio)):return False
  return True
 def _analyze_dynamics_enhanced(self,audio:np.ndarray,sample_rate:int)->Tuple[float,List[str]]:
  issues=[];mono=np.mean(audio,axis=0)if audio.ndim>1 else audio;rms=np.sqrt(np.mean(mono**2));peak=np.max(np.abs(mono));percentiles=np.percentile(np.abs(mono),[1,5,10,90,95,99]);p1,p5,p10,p90,p95,p99=percentiles;dr_main=20*np.log10(p99/(p1+1e-10));dr_mid=20*np.log10(p95/(p5+1e-10));dr_core=20*np.log10(p90/(p10+1e-10))
  if dr_main<6:issues.append("Rango dinámico muy limitado (over-compression)")
  elif dr_main>60:issues.append("Rango dinámico excesivo (posible ruido)")
  elif dr_core<8:issues.append("Núcleo dinámico comprimido")
  crest_factor=peak/(rms+1e-10)
  if crest_factor<1.2:issues.append("Factor cresta crítico (limitación severa)")
  elif crest_factor<1.5:issues.append("Factor cresta bajo (posible limitación)")
  loudness_rms=np.sqrt(np.mean(mono[np.abs(mono)>np.percentile(np.abs(mono),50)]**2));plr=peak/(loudness_rms+1e-10)
  if plr<2.0:issues.append("Relación pico-loudness baja")
  dynamic_score=min(100,max(0,(dr_main-6)*2.5+(dr_core-8)*1.5));dynamic_score*=min(1.0,crest_factor/1.5);return dynamic_score,issues
 @lru_cache(maxsize=32)
 def _get_fft_window(self,size:int,window_type:str="hann")->np.ndarray:
  if window_type=="hann":return np.hanning(size)
  elif window_type=="blackman":return np.blackman(size)
  return np.ones(size)
 def _analyze_spectrum_enhanced(self,audio:np.ndarray,sample_rate:int)->Tuple[Dict[str,float],List[str]]:
  issues=[];mono=np.mean(audio,axis=0)if audio.ndim>1 else audio;window=self._get_fft_window(self.config.fft_size);hop_size=int(self.config.fft_size*(1-self.config.overlap));spectrogram=[]
  for i in range(0,len(mono)-self.config.fft_size,hop_size):frame=mono[i:i+self.config.fft_size]*window;fft_frame=np.fft.rfft(frame,n=self.config.fft_size*2);spectrogram.append(np.abs(fft_frame))
  if not spectrogram:return {},"Audio demasiado corto para análisis espectral"
  magnitude=np.mean(spectrogram,axis=0);freqs=np.fft.rfftfreq(self.config.fft_size*2,1/sample_rate);bands_detailed={'sub_bass':(20,60),'bass':(60,120),'low_bass':(120,250),'low_mid':(250,500),'mid':(500,1000),'upper_mid':(1000,2000),'high_mid':(2000,4000),'presence':(4000,8000),'brilliance':(8000,12000),'air':(12000,sample_rate//2)};band_energies={}
  for band_name,(low,high)in bands_detailed.items():mask=(freqs>=low)&(freqs<=high);band_energies[band_name]=10*np.log10(np.mean(magnitude[mask]**2)+1e-12)if np.any(mask)else-80
  total_energy=np.mean([e for e in band_energies.values()if e>-70])
  for band,energy in band_energies.items():deviation=energy-total_energy;(issues.append(f"Exceso significativo en {band} (+{deviation:.1f}dB)")if deviation>8 else issues.append(f"Ligero exceso en {band} (+{deviation:.1f}dB)")if deviation>6 else issues.append(f"Déficit severo en {band} ({deviation:.1f}dB)")if deviation<-15 else issues.append(f"Déficit en {band} ({deviation:.1f}dB)")if deviation<-12 else None)
  spectral_variance=np.var([e for e in band_energies.values()if e>-70])
  if spectral_variance>50:issues.append("Alta varianza espectral - posibles resonancias")
  return band_energies,issues
 def _analyze_phase_coherence_enhanced(self,audio:np.ndarray)->Tuple[float,List[str]]:
  if audio.ndim!=2:return 100,[]
  issues=[];left,right=audio[0],audio[1];correlation=np.corrcoef(left,right)[0,1];correlation=0.0 if np.isnan(correlation)else correlation;mid=(left+right)/2;side=(left-right)/2;mid_energy=np.mean(mid**2);side_energy=np.mean(side**2);band_correlations={}
  for band_name,(low,high)in[('low',(20,250)),('mid',(250,2000)),('high',(2000,20000))]:
   try:sos=ss.butter(4,[low,high],btype='band',fs=44100,output='sos');left_filt=ss.sosfilt(sos,left);right_filt=ss.sosfilt(sos,right);band_corr=np.corrcoef(left_filt,right_filt)[0,1];band_correlations[band_name]=band_corr if not np.isnan(band_corr)else correlation
   except:band_correlations[band_name]=correlation
  if correlation<0.1:issues.append("Correlación crítica - posible inversión de fase")
  elif correlation<0.3:issues.append("Baja correlación estéreo (problemas de fase)")
  elif correlation>0.98:issues.append("Correlación excesiva (imagen estéreo muy limitada)")
  elif correlation>0.95:issues.append("Correlación muy alta (imagen estéreo limitada)")
  ms_ratio=10*np.log10(mid_energy/(side_energy+1e-12))
  if ms_ratio<-12:issues.append("Exceso severo de información lateral")
  elif ms_ratio<-6:issues.append("Exceso de información lateral")
  elif ms_ratio>25:issues.append("Imagen estéreo extremadamente estrecha")
  elif ms_ratio>20:issues.append("Imagen estéreo muy estrecha")
  correlation_score=correlation*100
  for band,corr in band_correlations.items():
   if corr<0.2:correlation_score*=0.9;issues.append(f"Problemas de fase en banda {band}")
  return correlation_score,issues
 def _analyze_distortion_enhanced(self,audio:np.ndarray,sample_rate:int)->Tuple[float,List[str]]:
  issues=[];mono=np.mean(audio,axis=0)if audio.ndim>1 else audio;clipping_threshold=0.99;near_clipping_threshold=0.95;clipped_samples=np.sum(np.abs(mono)>=clipping_threshold);near_clipped_samples=np.sum(np.abs(mono)>=near_clipping_threshold);clipping_ratio=clipped_samples/len(mono);near_clipping_ratio=near_clipped_samples/len(mono)
  if clipping_ratio>0.001:issues.append(f"Clipping detectado ({clipping_ratio*100:.3f}%)")
  elif near_clipping_ratio>0.01:issues.append(f"Samples cerca del clipping ({near_clipping_ratio*100:.2f}%)")
  kurt_value=kurtosis(mono)
  if kurt_value>15:issues.append("Curtosis muy alta (distorsión severa)")
  elif kurt_value>10:issues.append("Alta curtosis (posible distorsión)")
  elif kurt_value<1.5:issues.append("Curtosis baja (posible limitación excesiva)")
  harmonic_distortion=self._calculate_thd_enhanced(mono,sample_rate)
  if harmonic_distortion>5.0:issues.append(f"THD alto ({harmonic_distortion:.1f}%)")
  elif harmonic_distortion>2.0:issues.append(f"THD moderado ({harmonic_distortion:.1f}%)")
  imd_score=self._detect_intermodulation_distortion(mono,sample_rate)
  if imd_score>3.0:issues.append("Distorsión de intermodulación detectada")
  distortion_score=100;distortion_score-=min(40,clipping_ratio*4000);distortion_score-=min(20,max(0,kurt_value-3)*2);distortion_score-=min(20,harmonic_distortion*4);distortion_score-=min(10,imd_score*3);return max(0,distortion_score),issues
 def _analyze_lufs_precise(self,audio:np.ndarray,sample_rate:int)->Tuple[float,List[str]]:
  issues=[];audio_stereo=np.stack([audio,audio])if audio.ndim==1 else audio
  try:sos_hp=ss.butter(4,38,btype='high',fs=sample_rate,output='sos');audio_hp=ss.sosfilt(sos_hp,audio_stereo,axis=-1);sos_shelf=ss.butter(2,1500,btype='high',fs=sample_rate,output='sos');audio_weighted=ss.sosfilt(sos_shelf,audio_hp,axis=-1);mean_square_left=np.mean(audio_weighted[0]**2);mean_square_right=np.mean(audio_weighted[1]**2)if audio_weighted.shape[0]>1 else mean_square_left;lufs=-0.691+10*np.log10(mean_square_left+mean_square_right)
  except Exception as e:rms=np.sqrt(np.mean(audio_stereo**2));lufs=20*np.log10(rms+1e-12);issues.append("LUFS calculado con método simplificado")
  if lufs>-6:issues.append(f"LUFS muy alto ({lufs:.1f}) - riesgo de fatiga auditiva")
  elif lufs>-12:issues.append(f"LUFS alto ({lufs:.1f}) - considerar reducir nivel")
  elif lufs<-30:issues.append(f"LUFS muy bajo ({lufs:.1f}) - señal débil")
  elif lufs<-25:issues.append(f"LUFS bajo ({lufs:.1f}) - considerar aumentar nivel")
  windowed_loudness=[];window_size=int(sample_rate*3.0);hop_size=int(window_size*0.1)
  for i in range(0,len(audio_stereo[0])-window_size,hop_size):
   window=audio_stereo[:,i:i+window_size];window_rms=np.sqrt(np.mean(window**2))
   if window_rms>1e-6:windowed_loudness.append(20*np.log10(window_rms))
  if len(windowed_loudness)>2:
   lra=np.percentile(windowed_loudness,95)-np.percentile(windowed_loudness,10)
   if lra<3:issues.append(f"LRA muy bajo ({lra:.1f}) - audio muy comprimido")
   elif lra>20:issues.append(f"LRA muy alto ({lra:.1f}) - posible inconsistencia de nivel")
  return lufs,issues
 def _detect_digital_artifacts(self,audio:np.ndarray,sample_rate:int)->Tuple[float,List[str]]:
  issues=[];mono=np.mean(audio,axis=0)if audio.ndim>1 else audio;fft_data=np.abs(np.fft.rfft(mono));freqs=np.fft.rfftfreq(len(mono),1/sample_rate);nyquist=sample_rate/2;high_freq_mask=freqs>(nyquist*0.9)
  if np.any(high_freq_mask):high_freq_energy=np.mean(fft_data[high_freq_mask]**2);total_energy=np.mean(fft_data**2);aliasing_ratio=high_freq_energy/(total_energy+1e-12);(issues.append("Posible aliasing detectado")if aliasing_ratio>0.01 else None)
  if len(mono)>1000:quantized=np.round(mono*32768)/32768;quantization_error=mono-quantized;qe_energy=np.mean(quantization_error**2);(issues.append("Ruido de cuantización detectado")if qe_energy>1e-8 else None)
  diff_signal=np.diff(mono)
  if len(diff_signal)>0:click_threshold=np.std(diff_signal)*10;clicks=np.sum(np.abs(diff_signal)>click_threshold);(issues.append(f"Clicks/pops detectados ({clicks} eventos)")if clicks>len(mono)*0.0001 else None)
  artifact_score=100-len(issues)*15;return max(0,artifact_score),issues
 def _analyze_noise_floor(self,audio:np.ndarray,sample_rate:int)->Tuple[float,List[str]]:
  issues=[];mono=np.mean(audio,axis=0)if audio.ndim>1 else audio;abs_signal=np.abs(mono);noise_floor_estimate=np.percentile(abs_signal,10);signal_level=np.percentile(abs_signal,90);snr_db=20*np.log10(signal_level/noise_floor_estimate)if noise_floor_estimate>0 else 100;noise_bands={}
  for band_name,(low,high)in[('low',(20,250)),('mid',(250,2000)),('high',(2000,8000))]:
   try:sos=ss.butter(4,[low,high],btype='band',fs=sample_rate,output='sos');filtered=ss.sosfilt(sos,mono);band_noise=np.percentile(np.abs(filtered),10);noise_bands[band_name]=20*np.log10(band_noise+1e-12)
   except:noise_bands[band_name]=-80
  if snr_db<40:issues.append(f"SNR bajo ({snr_db:.1f}dB) - ruido audible")
  elif snr_db<50:issues.append(f"SNR moderado ({snr_db:.1f}dB)")
  for band,noise_level in noise_bands.items():
   if noise_level>-60:issues.append(f"Ruido elevado en banda {band}")
  hum_freqs=[50,60,100,120,150,180];fft_data=np.abs(np.fft.rfft(mono));freqs=np.fft.rfftfreq(len(mono),1/sample_rate)
  for hum_freq in hum_freqs:
   freq_idx=np.argmin(np.abs(freqs-hum_freq))
   if freq_idx<len(fft_data)and fft_data[freq_idx]>np.mean(fft_data[max(0,freq_idx-5):min(len(fft_data),freq_idx+5)])*5:
    issues.append(f"Posible hum a {hum_freq}Hz")
    break
  noise_score=min(100,max(0,snr_db-20));return noise_score,issues
 def _calculate_thd_enhanced(self,signal:np.ndarray,sample_rate:int)->float:
  try:
   fft_data=np.abs(np.fft.rfft(signal));freqs=np.fft.rfftfreq(len(signal),1/sample_rate)
   audible_mask=(freqs>=80)&(freqs<=2000)
   if not np.any(audible_mask):return 0.0
   fundamental_idx=np.argmax(fft_data[audible_mask]);fundamental_freq=freqs[audible_mask][fundamental_idx]
   fundamental_magnitude=fft_data[audible_mask][fundamental_idx];harmonic_power=0
   for harmonic in range(2,6):
    if fundamental_freq*harmonic<sample_rate/2:
     harmonic_power+=fft_data[np.argmin(np.abs(freqs-fundamental_freq*harmonic))]**2
   return min(100*np.sqrt(harmonic_power)/fundamental_magnitude,100)if fundamental_magnitude>0 else 0.0
  except:return 0.0
 def _detect_intermodulation_distortion(self,signal:np.ndarray,sample_rate:int)->float:
  try:
   fft_data=np.abs(np.fft.rfft(signal));freqs=np.fft.rfftfreq(len(signal),1/sample_rate)
   peaks,_=ss.find_peaks(fft_data,height=np.max(fft_data)*0.1)
   if len(peaks)<2:return 0.0
   peak_magnitudes=fft_data[peaks];sorted_indices=np.argsort(peak_magnitudes)[-2:]
   main_peaks=peaks[sorted_indices];f1,f2=freqs[main_peaks]
   imd_freqs=[abs(f1-f2),f1+f2,abs(2*f1-f2),abs(2*f2-f1)]
   imd_power=sum(fft_data[np.argmin(np.abs(freqs-imd_freq))]**2 for imd_freq in imd_freqs if imd_freq<sample_rate/2 and imd_freq>20)
   fundamental_power=np.sum(fft_data[main_peaks]**2)
   return min(100*np.sqrt(imd_power)/np.sqrt(fundamental_power),100)if fundamental_power>0 else 0.0
  except:return 0.0
 def _calculate_overall_score_enhanced(self,metricas:Dict[str,Any])->float:
  weights={'dynamic_range':0.25,'phase_coherence':0.15,'thd':0.20,'spectral_balance':0.15,'lufs_analysis':0.10,'artifacts':0.10,'noise_floor':0.05}
  total_score=0;total_weight=0
  for metric,weight in weights.items():
   if metric in metricas:
    value=metricas[metric]
    if isinstance(value,dict)and metric=='spectral_balance':
     score=max(0,100-np.var([v for v in value.values()if v>-70])*1.5)if value else 75
    elif isinstance(value,dict):
     score=75
    elif isinstance(value,(int,float))and metric=='lufs_analysis':
     score=max(0,100-abs(value-(-23))*3)
    elif isinstance(value,(int,float)):
     score=float(value)
    else:
     score=75
    score=max(0,min(100,score));total_score+=score*weight;total_weight+=weight
  final_score=total_score/max(total_weight,0.1)
  advanced_metrics=['artifacts','noise_floor','lufs_analysis']
  available_advanced=sum(1 for m in advanced_metrics if m in metricas);bonus=available_advanced*2
  return min(100,final_score+bonus)
 def _get_quality_level(self,score:float)->str:return("excelente+"if score>=92 else"excelente"if score>=88 else"muy bueno+"if score>=82 else"muy bueno"if score>=78 else"bueno+"if score>=72 else"bueno"if score>=68 else"aceptable+"if score>=62 else"aceptable"if score>=58 else"deficiente")
 def _generate_recommendations_enhanced(self,resultado:Dict[str,Any])->List[str]:
  recomendaciones=[];score=resultado["score"];problemas=resultado["problemas"];metricas=resultado["metricas"]
  if score<60:recomendaciones.append("Reprocesamiento crítico requerido - múltiples problemas detectados")
  elif score<75:recomendaciones.append("Mejoras recomendadas para calidad profesional")
  elif score<85:recomendaciones.append("Optimización menor para calidad premium")
  problema_keywords={"compression":"Reducir compresión/limitación para preservar dinámica natural","clipping":"Aplicar limitador suave y reducir ganancia de entrada","fase":"Verificar configuración estéreo y procesamiento Mid/Side","phase":"Revisar paneo y procesamiento de efectos estéreo","curtosis":"Reducir procesamiento agresivo que cause distorsión","snr":"Mejorar aislamiento acústico o usar gate/expander","hum":"Verificar conexiones eléctricas y filtrar frecuencias específicas","aliasing":"Aplicar filtro anti-aliasing antes de conversión","clicks":"Usar herramientas de de-clicking y revisar ediciones","lufs":"Ajustar nivel general según estándares de broadcast"}
  for keyword,recomendacion in problema_keywords.items():
   if any(keyword.lower()in p.lower()for p in problemas):
    if recomendacion not in recomendaciones:recomendaciones.append(recomendacion)
  if"dynamic_range"in metricas and metricas["dynamic_range"]<40:recomendaciones.append("Considerar masterización con menor compresión")
  if"artifacts"in metricas and metricas["artifacts"]<70:recomendaciones.append("Revisar cadena digital y configuraciones de conversión")
  if"noise_floor"in metricas and metricas["noise_floor"]<60:recomendaciones.append("Mejorar calidad de grabación o aplicar noise reduction")
  if any("binaural"in p.lower()or"estéreo"in p.lower()for p in problemas):recomendaciones.append("Contenido estéreo: verificar compatibilidad mono y balance L/R")
  return recomendaciones
 def _get_error_result(self,error_msg:str)->Dict[str,Any]:return{"score":0,"nivel":"error","problemas":[f"Error de análisis: {error_msg}"],"recomendaciones":["Verificar formato, contenido y integridad del audio"],"metricas":{},"processing_time":0.0}
class CorrectionStrategy:
 def __init__(self,mode:str=ModoProcesamiento.PROFESIONAL):self.mode=mode;self._correction_cache={};self._filter_cache={}
 @staticmethod
 def aplicar(audio:np.ndarray,metadata:Dict[str,Any])->np.ndarray:return CorrectionStrategy().aplicar_avanzado(audio,metadata)
 def aplicar_avanzado(self,audio:np.ndarray,metadata:Dict[str,Any])->np.ndarray:
  if not isinstance(audio,np.ndarray):return audio
  corrected=audio.copy();problemas=metadata.get("problemas",[]);score=metadata.get("score",100)
  if score>=80:return corrected
  try:
   if any("clipping"in p.lower()for p in problemas):corrected=self._soft_clip_correction_enhanced(corrected)
   if any("compression"in p.lower()or"dinámico"in p.lower()for p in problemas):corrected=self._dynamic_expansion_enhanced(corrected)
   if any("energía"in p or"exceso"in p.lower()or"déficit"in p.lower()for p in problemas):corrected=self._spectral_correction_enhanced(corrected,problemas)
   if any("fase"in p.lower()or"phase"in p.lower()for p in problemas):corrected=self._phase_correction_enhanced(corrected)
   if any("lufs"in p.lower()for p in problemas):corrected=self._level_optimization_enhanced(corrected,metadata)
   if any("click"in p.lower()or"pop"in p.lower()for p in problemas):corrected=self._declick_enhanced(corrected)
   return corrected
  except Exception as e:logger.warning(f"Error en corrección: {e}");return audio
 def _soft_clip_correction_enhanced(self,audio:np.ndarray)->np.ndarray:
  threshold_hard=0.99;threshold_soft=0.95
  for threshold,strength in[(threshold_hard,0.3),(threshold_soft,0.1)]:
   mask=np.abs(audio)>threshold
   if np.any(mask):audio[mask]=np.tanh(audio[mask]*(1.0-strength))*threshold
  return audio
 def _dynamic_expansion_enhanced(self,audio:np.ndarray)->np.ndarray:
  for channel in range(audio.shape[0]if audio.ndim>1 else 1):
   signal=audio[channel]if audio.ndim>1 else audio
   channel_idx=None if audio.ndim==1 else channel
   window_size=max(1024,min(int(len(signal)*0.01),8192))
   rms_windowed=[np.sqrt(np.mean(signal[i:i+window_size]**2))for i in range(0,len(signal),window_size//4)if len(signal[i:i+window_size])>0]
   if rms_windowed:
    avg_rms=np.mean(rms_windowed);rms_variation=np.std(rms_windowed)
    compression_factor=max(0.1,min(0.3,1.0-rms_variation/avg_rms))if avg_rms>0 else 0.1
   else:
    compression_factor=0.1
   expansion_factor=1.0+compression_factor*0.5;signal_expanded=signal*expansion_factor
   max_val=np.max(np.abs(signal_expanded))
   if max_val>0.95:signal_expanded=signal_expanded*0.90/max_val
   if channel_idx is not None:audio[channel_idx]=signal_expanded
   else:audio=signal_expanded
  return audio
 def _spectral_correction_enhanced(self,audio:np.ndarray,problemas:List[str])->np.ndarray:
  corrected=audio.copy();eq_adjustments={}
  for problema in problemas:
   if"sub"in problema.lower()and"exceso"in problema.lower():eq_adjustments['sub_bass']=-2.0
   elif"bass"in problema.lower()and"exceso"in problema.lower():eq_adjustments['bass']=-1.5
   elif"bass"in problema.lower()and"déficit"in problema.lower():eq_adjustments['bass']=+1.0
   elif"high"in problema.lower()and"exceso"in problema.lower():eq_adjustments['high']=-1.5
   elif"high"in problema.lower()and"déficit"in problema.lower():eq_adjustments['high']=+1.0
   elif"presence"in problema.lower()and"exceso"in problema.lower():eq_adjustments['presence']=-1.0
  try:
   from scipy.signal import butter,sosfilt
   for band,gain_db in eq_adjustments.items():
    if abs(gain_db)>0.1:
     band_freqs={'sub_bass':(20,60),'bass':(60,250),'low_mid':(250,500),'mid':(500,2000),'high_mid':(2000,4000),'presence':(4000,8000),'high':(8000,16000)}
     low,high=band_freqs[band]if band in band_freqs else(None,None)
     sos=butter(2,[low,high],btype='band',fs=44100,output='sos')if low and high else None
     if sos is not None:
      for channel in range(corrected.shape[0]):corrected[channel]=corrected[channel]+sosfilt(sos,corrected[channel])*(10**(gain_db/20))*0.3
  except Exception as e:logger.warning(f"Error en corrección espectral: {e}")
  return corrected
 def _phase_correction_enhanced(self,audio:np.ndarray)->np.ndarray:
  if audio.ndim!=2:return audio
  corrected=audio.copy();left,right=corrected[0],corrected[1]
  try:
   from scipy.signal import butter,sosfilt
   bands=[(80,250),(250,2000),(2000,8000)]
   for low,high in bands:
    sos=butter(4,[low,high],btype='band',fs=44100,output='sos')
    left_band=sosfilt(sos,left);right_band=sosfilt(sos,right)
    correlation=np.corrcoef(left_band,right_band)[0,1]
    if not np.isnan(correlation)and correlation<0.1:
     mid=(left_band+right_band)/2;correction_strength=0.2
     left_band=left_band*(1-correction_strength)+mid*correction_strength
     right_band=right_band*(1-correction_strength)+mid*correction_strength
     corrected[0]=corrected[0]+left_band*0.1;corrected[1]=corrected[1]+right_band*0.1
  except Exception as e:logger.warning(f"Error en corrección de fase: {e}")
  return corrected
 def _level_optimization_enhanced(self,audio:np.ndarray,metadata:Dict[str,Any])->np.ndarray:
  lufs_target=-23.0;rms=np.sqrt(np.mean(audio**2));current_lufs_estimate=20*np.log10(rms+1e-12);lufs_adjustment=np.clip(lufs_target-current_lufs_estimate,-6,6)
  if abs(lufs_adjustment)>0.5:gain_linear=10**(lufs_adjustment/20);audio_adjusted=audio*gain_linear;max_val=np.max(np.abs(audio_adjusted));return audio_adjusted*0.90/max_val if max_val>0.95 else audio_adjusted
  return audio
 def _declick_enhanced(self,audio:np.ndarray)->np.ndarray:
  corrected=audio.copy()
  for channel in range(corrected.shape[0]):
   signal=corrected[channel];diff_signal=np.diff(signal);threshold=np.std(diff_signal)*8
   click_indices=np.where(np.abs(diff_signal)>threshold)[0]
   for idx in click_indices:
    if 2<=idx<len(signal)-2:
     signal[idx]=(signal[idx-1]+signal[idx+1])/2
     for offset in[-1,1]:
      if 0<=idx+offset<len(signal):signal[idx+offset]=signal[idx+offset]*0.8+signal[idx]*0.2
   corrected[channel]=signal
  return corrected
class AuroraQualityPipeline:
 def __init__(self,mode:str=ModoProcesamiento.PROFESIONAL):
  self.mode=mode;self.analyzer=CarmineAnalyzerBasic();self.estrategia=CorrectionStrategy(mode);self._processing_stats={'total_processed':0,'avg_score':0,'processing_time':0,'corrections_applied':0,'score_improvements':[]}
  if mode==ModoProcesamiento.PROFESIONAL:self.analyzer.config.enable_artifact_detection=True;self.analyzer.config.enable_noise_analysis=True;self.analyzer.config.enable_advanced_lufs=True
  elif mode==ModoProcesamiento.MASTERING:self.analyzer.config.quality_threshold=90.0;self.analyzer.config.fft_size=8192
  elif mode==ModoProcesamiento.REALTIME:self.analyzer.config.fft_size=2048;self.analyzer.config.enable_artifact_detection=False
 def procesar(self,audio_func:Callable)->Tuple[np.ndarray,Dict[str,Any]]:
  start_time=time.time()
  try:
   audio=audio_func();audio_original=audio.copy();audio=self.validar_y_normalizar(audio)
   resultado=self.analyzer.evaluar_avanzado(audio)if self.mode==ModoProcesamiento.DEBUG else CarmineAnalyzerBasic.evaluar(audio)
   score_inicial=resultado["score"]
   if resultado["score"]<85:
    audio_corregido=self.estrategia.aplicar_avanzado(audio,resultado)if self.mode==ModoProcesamiento.PROFESIONAL else CorrectionStrategy.aplicar(audio,resultado)
    if not np.array_equal(audio,audio_corregido):
     resultado_post=CarmineAnalyzerBasic.evaluar(audio_corregido)
     if resultado_post["score"]>score_inicial:
      audio=audio_corregido;resultado=resultado_post
      self._processing_stats['corrections_applied']+=1
      self._processing_stats['score_improvements'].append(resultado["score"]-score_inicial)
   processing_time=time.time()-start_time;self._update_stats(resultado,processing_time)
   resultado["processing_metadata"]={"mode":self.mode,"original_shape":audio_original.shape,"final_shape":audio.shape,"score_inicial":score_inicial,"score_final":resultado["score"],"corrections_applied":score_inicial!=resultado["score"],"processing_time":processing_time}
   return audio,resultado
  except Exception as e:logger.error(f"Error en pipeline: {e}");audio=audio_func();return audio,{"score":50,"nivel":"error","problemas":[str(e)],"processing_time":time.time()-start_time}
 def validar_y_normalizar(self,signal:np.ndarray)->np.ndarray:
  if not isinstance(signal,np.ndarray):raise ValueError("❌ Señal inválida: debe ser un arreglo numpy")
  if np.any(np.isnan(signal))or np.any(np.isinf(signal)):logger.warning("Valores NaN/Inf detectados, limpiando...");signal=np.nan_to_num(signal,nan=0.0,posinf=0.0,neginf=0.0)
  if signal.ndim==1:signal=np.stack([signal,signal])
  elif signal.ndim==2:
   signal=signal.T if signal.shape[0]>signal.shape[1]else signal
   if signal.shape[0]>2:signal=np.stack([np.mean(signal[:2],axis=0),np.mean(signal[:2],axis=0)])
  else:raise ValueError("❌ Señal inválida: debe ser mono [N] o estéreo [2, N]")
  max_val=np.max(np.abs(signal))
  if max_val>0:
   target_level=0.99 if self.mode==ModoProcesamiento.MASTERING else 0.95 if self.mode==ModoProcesamiento.PROFESIONAL else 0.90
   if max_val>target_level or max_val<target_level*0.1:signal=signal*(target_level/max_val)
  return np.clip(signal,-1.0,1.0)
 def _update_stats(self,resultado:Dict[str,Any],processing_time:float):
  self._processing_stats['total_processed']+=1;current_avg=self._processing_stats['avg_score'];total=self._processing_stats['total_processed'];new_score=resultado.get('score',0);self._processing_stats['avg_score']=(current_avg*(total-1)+new_score)/total;current_time_avg=self._processing_stats['processing_time'];self._processing_stats['processing_time']=(current_time_avg*(total-1)+processing_time)/total
  if len(self._processing_stats['score_improvements'])>100:self._processing_stats['score_improvements']=self._processing_stats['score_improvements'][-100:]
 def get_stats(self)->Dict[str,Any]:
  stats=self._processing_stats.copy()
  if stats['score_improvements']:stats['avg_improvement']=np.mean(stats['score_improvements']);stats['max_improvement']=np.max(stats['score_improvements']);stats['improvement_rate']=len(stats['score_improvements'])/max(1,stats['total_processed'])
  else:stats['avg_improvement']=0;stats['max_improvement']=0;stats['improvement_rate']=0
  stats['mode']=self.mode;stats['config']={'quality_threshold':self.analyzer.config.quality_threshold,'fft_size':self.analyzer.config.fft_size,'advanced_features_enabled':self.analyzer.config.enable_artifact_detection};return stats
 def reset_stats(self):mode=self.mode;self._processing_stats={'total_processed':0,'avg_score':0,'processing_time':0,'corrections_applied':0,'score_improvements':[]};self.mode=mode
def evaluar_calidad_audio(audio:np.ndarray,sample_rate:int=44100,modo_avanzado:bool=False)->Dict[str,Any]:
 analyzer=CarmineAnalyzerBasic()
 if modo_avanzado:
  analyzer.config.enable_artifact_detection=True
  analyzer.config.enable_noise_analysis=True
  analyzer.config.enable_advanced_lufs=True
 return analyzer.evaluar_avanzado(audio,sample_rate)
def corregir_audio_automaticamente(audio:np.ndarray,modo:str=ModoProcesamiento.PROFESIONAL,forzar_correccion:bool=False)->Tuple[np.ndarray,Dict[str,Any]]:
 pipeline=AuroraQualityPipeline(mode=modo)
 if forzar_correccion:
  original_threshold=pipeline.analyzer.config.quality_threshold
  pipeline.analyzer.config.quality_threshold=90.0
  result=pipeline.procesar(lambda:audio)
  pipeline.analyzer.config.quality_threshold=original_threshold
  return result
 else:
  return pipeline.procesar(lambda:audio)
def validar_audio_estereo(audio:np.ndarray,modo:str=ModoProcesamiento.PROFESIONAL)->np.ndarray:pipeline=AuroraQualityPipeline(mode=modo);return pipeline.validar_y_normalizar(audio)
def analizar_comparativo(audio_original:np.ndarray,audio_procesado:np.ndarray,sample_rate:int=44100)->Dict[str,Any]:resultado_original=evaluar_calidad_audio(audio_original,sample_rate,modo_avanzado=True);resultado_procesado=evaluar_calidad_audio(audio_procesado,sample_rate,modo_avanzado=True);return{"original":resultado_original,"procesado":resultado_procesado,"mejora_score":resultado_procesado["score"]-resultado_original["score"],"problemas_resueltos":len(resultado_original["problemas"])-len(resultado_procesado["problemas"]),"nuevos_problemas":[p for p in resultado_procesado["problemas"]if p not in resultado_original["problemas"]],"timestamp":time.time()}
if __name__=="__main__":
 print("🔍 Aurora Quality Pipeline V2.0 - Sistema de Validación Optimizado");print("="*70);sample_rate=44100;duration=3.0;t=np.linspace(0,duration,int(sample_rate*duration));left_channel=(0.7*np.sin(2*np.pi*440*t)+0.2*np.sin(2*np.pi*880*t)+0.1*np.random.normal(0,0.05,len(t))+0.05*np.sin(2*np.pi*50*t));right_channel=(0.6*np.sin(2*np.pi*444*t)+0.15*np.sin(2*np.pi*888*t)+0.1*np.random.normal(0,0.05,len(t)));left_channel=np.clip(left_channel*1.2,-1.0,1.0);right_channel=np.clip(right_channel*1.15,-1.0,1.0);test_audio=np.stack([left_channel,right_channel]);print(f"📊 Analizando audio de prueba (duración: {duration}s)...")
 print(f"\n🔧 Test 1: Análisis básico");start_time=time.time();resultado_basico=evaluar_calidad_audio(test_audio,sample_rate);tiempo_basico=time.time()-start_time;print(f"   ✅ Score: {resultado_basico['score']:.1f}/100 ({resultado_basico['nivel']})");print(f"   ⏱️ Tiempo: {tiempo_basico:.3f}s");print(f"   🔍 Problemas detectados: {len(resultado_basico['problemas'])}")
 print(f"\n🚀 Test 2: Análisis avanzado");start_time=time.time();resultado_avanzado=evaluar_calidad_audio(test_audio,sample_rate,modo_avanzado=True);tiempo_avanzado=time.time()-start_time;print(f"   ✅ Score: {resultado_avanzado['score']:.1f}/100 ({resultado_avanzado['nivel']})");print(f"   ⏱️ Tiempo: {tiempo_avanzado:.3f}s");print(f"   🔍 Problemas detectados: {len(resultado_avanzado['problemas'])}");print(f"   📈 Métricas adicionales: {len(resultado_avanzado['metricas'])} tipos")
 if resultado_avanzado['problemas']:print(f"   📋 Ejemplos de problemas:");[print(f"      • {problema}")for problema in resultado_avanzado['problemas'][:3]]
 print(f"\n🔧 Test 3: Corrección automática")
 try:start_time=time.time();audio_corregido,analisis_corregido=corregir_audio_automaticamente(test_audio,modo=ModoProcesamiento.PROFESIONAL);tiempo_correccion=time.time()-start_time;mejora=analisis_corregido['score']-resultado_avanzado['score'];print(f"   ✅ Score después de corrección: {analisis_corregido['score']:.1f}/100");print(f"   📈 Mejora: {mejora:+.1f} puntos");print(f"   ⏱️ Tiempo total: {tiempo_correccion:.3f}s");(print(f"   🛠️ Correcciones aplicadas exitosamente")if"processing_metadata"in analisis_corregido and analisis_corregido["processing_metadata"].get("corrections_applied",False)else print(f"   ℹ️ No se requirieron correcciones"))
 except Exception as e:print(f"   ⚠️ Error en corrección: {e}")
 print(f"\n📊 Test 4: Análisis comparativo")
 try:comparacion=analizar_comparativo(test_audio,audio_corregido,sample_rate);print(f"   📈 Mejora total: {comparacion['mejora_score']:+.1f} puntos");print(f"   🔧 Problemas resueltos: {comparacion['problemas_resueltos']}");(print(f"   ⚠️ Nuevos problemas: {len(comparacion['nuevos_problemas'])}")if comparacion['nuevos_problemas']else None)
 except Exception as e:print(f"   ⚠️ Error en comparación: {e}")
 print(f"\n⚙️ Test 5: Diferentes modos de procesamiento");modos=[ModoProcesamiento.PROFESIONAL,ModoProcesamiento.MASTERING,ModoProcesamiento.REALTIME]
 for modo in modos:
  try:start_time=time.time();pipeline=AuroraQualityPipeline(mode=modo);_,resultado_modo=pipeline.procesar(lambda:test_audio);tiempo_modo=time.time()-start_time;print(f"   {modo}: Score {resultado_modo['score']:.1f}/100 ({tiempo_modo:.3f}s)")
  except Exception as e:print(f"   {modo}: Error - {e}")
 print(f"\n📈 Mejoras implementadas:");[print(f"   ✅ {mejora}")for mejora in["Análisis LUFS más preciso","Detección de artefactos digitales","Análisis de ruido de fondo","Correcciones más musicales","Mejor performance y caching","Métricas más detalladas"]];print(f"\n🚀 Aurora Quality Pipeline V2.0 listo!");print(f"💎 Análisis técnico de calidad profesional optimizado!")
