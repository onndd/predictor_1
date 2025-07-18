"""
Colab Health Check and Setup Utilities
Bu dosya Colab notebook'undan import edilerek kullanılır.
"""

import os
import sys
import subprocess
import torch
import traceback
import warnings
from typing import Optional, Dict, Any, List
import yaml
from typing import Tuple

# Uyarıları bastır
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class ColabHealthCheck:
    """Colab ortamı için sağlık kontrolü ve kurulum yardımcı sınıfı"""
    
    def __init__(self):
        self.project_dir = "/content/predictor_1"
        self.repo_url = "https://github.com/onndd/predictor_1.git"
        self.gpu_info = None
        self.system_info = {}
        self.config: Dict[str, Any] = {}
        self.db_path: str = ""

    def _load_config(self) -> bool:
        """Proje yapılandırmasını yükler."""
        try:
            # Add src to path to find settings
            src_path = os.path.join(self.project_dir, "src")
            if src_path not in sys.path:
                sys.path.insert(0, src_path)

            from config.settings import CONFIG, DATABASE_PATH
            self.config = CONFIG
            self.db_path = os.path.join(self.project_dir, DATABASE_PATH)
            print("✅ Yapılandırma dosyası başarıyla yüklendi.")
            return True
        except Exception as e:
            print(f"❌ Yapılandırma yüklenemedi: {e}")
            return False

    def check_gpu_availability(self):
        """GPU durumunu kontrol et"""
        try:
            if torch.cuda.is_available():
                self.gpu_info = {
                    'available': True,
                    'device_name': torch.cuda.get_device_name(0),
                    'memory_total': torch.cuda.get_device_properties(0).total_memory,
                    'memory_allocated': torch.cuda.memory_allocated(),
                    'memory_free': torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                }
                print(f"✅ GPU bulundu: {self.gpu_info['device_name']}")
                print(f"💾 GPU Memory: {self.gpu_info['memory_total'] / 1024**3:.1f} GB")
                return True
            else:
                self.gpu_info = {'available': False}
                print("⚠️ GPU bulunamadı. CPU ile devam edilecek.")
                return False
        except Exception as e:
            print(f"❌ GPU kontrolü sırasında hata: {e}")
            return False
    
    def check_python_environment(self):
        """Python ortamını kontrol et"""
        try:
            python_version = sys.version_info
            print(f"🐍 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
            
            # Gerekli kütüphaneleri kontrol et
            required_packages = ['torch', 'numpy', 'sqlite3', 'yaml']
            missing_packages = []
            
            for package in required_packages:
                try:
                    __import__(package)
                    print(f"✅ {package} yüklü")
                except ImportError:
                    missing_packages.append(package)
                    print(f"❌ {package} eksik")
            
            return len(missing_packages) == 0
            
        except Exception as e:
            print(f"❌ Python ortamı kontrolü sırasında hata: {e}")
            return False
    
    def setup_environment_safe(self):
        """Güvenli ortam kurulumu"""
        try:
            print("🛠️ Güvenli ortam kurulumu başlatılıyor...")
            
            # Proje dizinini kontrol et
            if os.path.exists(self.project_dir):
                print("📁 Proje zaten mevcut. Güncelleniyor...")
                os.chdir(self.project_dir)
                try:
                    result = subprocess.run(["git", "pull", "origin", "main"], 
                                          check=True, capture_output=True, text=True, timeout=60)
                    print("✅ Git pull başarılı")
                except subprocess.TimeoutExpired:
                    print("⚠️ Git pull zaman aşımı - devam ediliyor")
                except subprocess.CalledProcessError as e:
                    print(f"⚠️ Git pull hatası: {e.stderr}")
                    print("🔄 Repo'yu yeniden klonlama...")
                    os.chdir("/content")
                    subprocess.run(["rm", "-rf", "predictor_1"], check=False)
                    subprocess.run(["git", "clone", self.repo_url], check=True, timeout=120)
                    os.chdir(self.project_dir)
            else:
                print("📥 GitHub'dan proje klonlanıyor...")
                os.chdir("/content")
                subprocess.run(["git", "clone", self.repo_url], check=True, timeout=120)
                os.chdir(self.project_dir)
            
            # Python path'i güncelle
            src_path = os.path.join(self.project_dir, "src")
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
                print(f"✅ {src_path} Python path'e eklendi")
            
            # Requirements kurulumu
            requirements_path = os.path.join(self.project_dir, 'requirements_enhanced.txt')
            if os.path.exists(requirements_path):
                print("📦 Gerekli kütüphaneler yükleniyor...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path], 
                                        timeout=300)
                    print("✅ Kütüphaneler başarıyla yüklendi")
                except subprocess.TimeoutExpired:
                    print("⚠️ Kütüphane kurulumu zaman aşımı")
                    return False
                except subprocess.CalledProcessError as e:
                    print(f"⚠️ Kütüphane kurulumu hatası: {e}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Ortam kurulumu sırasında hata: {e}")
            traceback.print_exc()
            return False
    
    def check_database_status(self):
        """Veritabanı durumunu kontrol et"""
        try:
            db_path = os.path.join(self.project_dir, "data", "jetx_data.db")
            
            if os.path.exists(db_path):
                file_size = os.path.getsize(db_path)
                print(f"📊 Database bulundu: {file_size / 1024:.1f} KB")
                
                # Basit SQLite kontrolü
                import sqlite3
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM jetx_results")
                count = cursor.fetchone()[0]
                conn.close()
                
                print(f"📈 Kayıt sayısı: {count}")
                return count >= 500  # Minimum 500 kayıt gerekli
            else:
                print("⚠️ Database bulunamadı - örnek veri oluşturulacak")
                return False
                
        except Exception as e:
            print(f"❌ Database kontrolü sırasında hata: {e}")
            return False
    
    def _validate_config(self) -> List[str]:
        """Yapılandırma dosyasını doğrular."""
        errors = []
        if not self.config:
            errors.append("Yapılandırma (config) boş veya yüklenemedi.")
            return errors

        # HPO arama uzayını doğrula
        hpo_config = self.config.get('hpo_search_space', {})
        for model, space in hpo_config.items():
            if not isinstance(space, dict): continue
            for param, values in space.items():
                if isinstance(values, dict) and 'low' in values and 'high' in values:
                    if values['low'] >= values['high']:
                        errors.append(f"HPO '{model} -> {param}': 'low' ({values['low']}) >= 'high' ({values['high']}).")

        # Eğitim profillerini doğrula
        profiles = self.config.get('aggressive_training_profiles', {})
        for model, profile in profiles.items():
            if 'sequence_length' not in profile:
                errors.append(f"Profil '{model}': 'sequence_length' eksik.")
        
        return errors

    def _check_data_compatibility(self) -> List[str]:
        """Veri sayısı ile sequence_length uyumluluğunu kontrol eder."""
        errors = []
        if not self.config or not self.db_path or not os.path.exists(self.db_path):
            errors.append("Veri uyumluluk kontrolü için config veya veritabanı bulunamadı.")
            return errors
        
        try:
            conn = sqlite3.connect(self.db_path)
            data_count = conn.execute("SELECT COUNT(*) FROM jetx_results").fetchone()[0]
            conn.close()

            profiles = self.config.get('aggressive_training_profiles', {})
            for model, profile in profiles.items():
                seq_len = profile.get('sequence_length')
                if seq_len and data_count < seq_len * 1.5:
                    errors.append(f"Veri '{model}': Gerekli veri ({int(seq_len * 1.5)}) > mevcut veri ({data_count}).")
            return errors
        except Exception as e:
            errors.append(f"Veritabanı uyumluluk kontrol hatası: {e}")
            return errors

    def check_ngrok_tunnel(self, authtoken: str) -> Tuple[bool, str]:
        """ngrok tünelini proaktif olarak test eder."""
        print("\n🔗 ngrok Tünel Bağlantısı Test Ediliyor...")
        try:
            from pyngrok import ngrok, conf
            
            # Kill any existing ngrok processes to ensure a clean start
            ngrok.kill()
            
            # Set authtoken
            ngrok.set_auth_token(authtoken)
            
            # Attempt to open a tunnel
            print("  - Geçici tünel oluşturuluyor...")
            http_tunnel = ngrok.connect(5000, "http")
            public_url = http_tunnel.public_url
            print(f"  - Tünel başarıyla oluşturuldu: {public_url}")
            
            # Close the tunnel immediately
            ngrok.disconnect(public_url)
            print("  - Geçici tünel kapatıldı.")
            
            return True, "ngrok bağlantısı başarılı."
            
        except Exception as e:
            error_message = f"ngrok hatası: {e}"
            print(f"  - ❌ {error_message}")
            # Ensure all ngrok processes are killed on failure
            try:
                ngrok.kill()
            except Exception:
                pass
            return False, error_message

    def run_pre_training_checks(self, ngrok_authtoken: Optional[str] = None):
        """Eğitim öncesi tüm kontrolleri yap"""
        print("🔍 Eğitim öncesi sistem kontrolleri başlatılıyor...")
        print("=" * 60)
        
        # Temel kurulum ve kontroller
        checks = {
            'GPU': self.check_gpu_availability(),
            'Python Ortamı': self.check_python_environment(),
            'Proje Kurulumu': self.setup_environment_safe(),
            'Veritabanı Durumu': self.check_database_status(),
            'Yapılandırma Yükleme': self._load_config()
        }

        # ngrok kontrolü (eğer token verilmişse)
        if ngrok_authtoken:
            ngrok_ok, _ = self.check_ngrok_tunnel(ngrok_authtoken)
            checks['ngrok Bağlantısı'] = ngrok_ok

        # Gelişmiş kontroller (eğer temel kurulum başarılıysa)
        if all(checks.values()):
            config_errors = self._validate_config()
            if config_errors:
                checks['Yapılandırma Doğrulama'] = False
                print("\n❌ Yapılandırma Hataları:")
                for err in config_errors: print(f"  - {err}")
            else:
                checks['Yapılandırma Doğrulama'] = True

            data_errors = self._check_data_compatibility()
            if data_errors:
                checks['Veri Uyumluluğu'] = False
                print("\n❌ Veri Uyumluluk Hataları:")
                for err in data_errors: print(f"  - {err}")
            else:
                checks['Veri Uyumluluğu'] = True
        
        print("\n📋 Kontrol Sonuçları:")
        for check_name, result in checks.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"  {check_name.upper()}: {status}")
        
        all_passed = all(checks.values())
        
        if all_passed:
            print("\n🎉 Tüm kontroller başarılı! Eğitim başlatılabilir.")
        else:
            print("\n⚠️ Bazı kontroller başarısız. Lütfen hataları düzeltin.")
            
        return all_passed
    
    def get_recommended_settings(self):
        """GPU ve sistem durumuna göre önerilen ayarları döndür"""
        settings = {
            'batch_size': 16,
            'epochs': 30,
            'chunk_size': 500,
            'n_trials': 10  # HPO için
        }
        
        if self.gpu_info and self.gpu_info['available']:
            memory_gb = self.gpu_info['memory_total'] / 1024**3
            
            if memory_gb >= 12:  # 12GB+ GPU
                settings.update({
                    'batch_size': 32,
                    'epochs': 50,
                    'chunk_size': 1000,
                    'n_trials': 20
                })
            elif memory_gb >= 8:  # 8GB+ GPU
                settings.update({
                    'batch_size': 24,
                    'epochs': 40,
                    'chunk_size': 750,
                    'n_trials': 15
                })
            # Düşük memory için varsayılan ayarlar
            
        print(f"🎯 Önerilen ayarlar: {settings}")
        return settings

def safe_import_with_fallback(module_name: str):
    """Güvenli module import'u - hata durumunda None döndür"""
    try:
        return __import__(module_name)
    except ImportError as e:
        print(f"⚠️ {module_name} import edilemedi: {e}")
        return None

# Kullanım örneği
if __name__ == "__main__":
    health_checker = ColabHealthCheck()
    if health_checker.run_pre_training_checks():
        print("✅ Sistem eğitim için hazır!")
        settings = health_checker.get_recommended_settings()
    else:
        print("❌ Lütfen hataları düzeltin ve tekrar deneyin.")
