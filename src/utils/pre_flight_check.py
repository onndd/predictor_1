#!/usr/bin/env python3
"""
Pre-Flight Check Script for JetX Prediction System

Bu script, Colab'da veya başka bir ortamda uzun süreli bir eğitim başlatmadan önce
proje yapılandırmasını, veri bütünlüğünü ve bağımlılıkları doğrulamak için kullanılır.
Olası hataları erkenden tespit ederek zaman kaybını önlemeyi amaçlar.
"""

import os
import sys
import sqlite3
import importlib.util
from typing import Dict, Any, List

# --- Dinamik Import ve Konfigürasyon Yükleme ---
CONFIG: Dict[str, Any] = {}
DATABASE_PATH: str = "data/jetx_data.db"  # Varsayılan fallback
YAML_ERROR: str | None = None

try:
    import yaml
    # Proje kök dizinini path'e ekle
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.config.settings import CONFIG as loaded_config, DATABASE_PATH as loaded_db_path
    CONFIG = loaded_config
    DATABASE_PATH = loaded_db_path
except ModuleNotFoundError:
    YAML_ERROR = "Kritik Bağımlılık Eksik: 'PyYAML' kütüphanesi bulunamadı. `config.yaml` kontrol edilemiyor. Lütfen 'pip install PyYAML' komutunu çalıştırın."
except ImportError:
    YAML_ERROR = "HATA: src.config.settings modülü bulunamadı. Lütfen script'i proje kök dizininden çalıştırdığınızdan emin olun."


# --- Renk Kodları ve Simgeler ---
class Style:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

ICON_OK = f"{Style.GREEN}✅{Style.END}"
ICON_WARN = f"{Style.YELLOW}⚠️{Style.END}"
ICON_FAIL = f"{Style.RED}❌{Style.END}"
ICON_INFO = f"{Style.BLUE}ℹ️{Style.END}"

def print_check(message: str, status: bool):
    """Kontrol sonucunu formatlı bir şekilde yazdırır."""
    icon = ICON_OK if status else ICON_FAIL
    print(f"{icon} {message}")

def validate_hpo_space(config: Dict[str, Any]) -> List[str]:
    """HPO arama uzayını daha detaylı doğrular."""
    if not config:
        return []
    errors = []
    hpo_config = config.get('hpo_search_space', {})
    if not isinstance(hpo_config, dict):
        return ["'hpo_search_space' bir sözlük (dictionary) olmalı."]

    for model, space in hpo_config.items():
        if model == 'n_trials': continue
        if not isinstance(space, dict):
            errors.append(f"HPO '{model}': Arama uzayı bir sözlük (dictionary) olmalı.")
            continue
        for param, values in space.items():
            if not isinstance(values, dict):
                errors.append(f"HPO '{model} -> {param}': Parametre ayarları bir sözlük olmalı.")
                continue
            if 'type' not in values:
                errors.append(f"HPO '{model} -> {param}': 'type' (categorical, float, int) belirtilmemiş.")
                continue
            
            param_type = values['type']
            if param_type == 'float' or param_type == 'int':
                if 'low' not in values or 'high' not in values:
                    errors.append(f"HPO '{model} -> {param}': '{param_type}' tipi için 'low' ve 'high' gereklidir.")
                elif not isinstance(values['low'], (int, float)) or not isinstance(values['high'], (int, float)):
                     errors.append(f"HPO '{model} -> {param}': 'low' ve 'high' sayısal değerler olmalıdır.")
                elif values['low'] >= values['high']:
                    errors.append(f"HPO '{model} -> {param}': 'low' ({values['low']}) değeri 'high' ({values['high']}) değerinden büyük veya eşit olamaz.")
            elif param_type == 'categorical':
                if 'choices' not in values or not isinstance(values['choices'], list):
                    errors.append(f"HPO '{model} -> {param}': 'categorical' tipi için 'choices' listesi gereklidir.")
    return errors

def validate_training_profiles(config: Dict[str, Any]) -> List[str]:
    """Eğitim profillerini doğrular."""
    if not config:
        return []
    errors = []
    profiles = config.get('aggressive_training_profiles', {})
    for model, profile in profiles.items():
        if 'sequence_length' not in profile:
            errors.append(f"Eğitim Profili '{model}': 'sequence_length' parametresi eksik.")
        if 'train_params' not in profile or 'epochs' not in profile['train_params']:
            errors.append(f"Eğitim Profili '{model}': 'train_params' altında 'epochs' parametresi eksik.")
    return errors

def check_data_compatibility(config: Dict[str, Any]) -> List[str]:
    """Veritabanı boyutu ile sequence_length uyumluluğunu kontrol eder."""
    if not config:
        return []
    errors = []
    try:
        if not os.path.exists(DATABASE_PATH):
            errors.append(f"Veritabanı dosyası bulunamadı: {DATABASE_PATH}")
            return errors

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM jetx_results")
        data_count = cursor.fetchone()[0]
        conn.close()

        print(f"{ICON_INFO} Veritabanında {data_count} kayıt bulundu.")

        min_data_multiplier = 1.5  # sequence_length'in en az 1.5 katı veri olmalı
        profiles = config.get('aggressive_training_profiles', {})
        for model, profile in profiles.items():
            seq_len = profile.get('sequence_length')
            if seq_len:
                required_data = int(seq_len * min_data_multiplier)
                if data_count < required_data:
                    errors.append(f"Veri Uyumluluğu '{model}': Gerekli veri sayısı ({required_data}) mevcut veriden ({data_count}) fazla. 'sequence_length'={seq_len}.")

    except Exception as e:
        errors.append(f"Veritabanı kontrolü sırasında hata: {e}")
    return errors

def check_dependencies() -> List[str]:
    """Gerekli kütüphanelerin yüklü olup olmadığını kontrol eder."""
    errors = []
    required_packages = [
        'torch', 'optuna', 'mlflow', 'streamlit',
        'sklearn', 'pandas', 'numpy', 'yaml'
    ]
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            errors.append(f"Bağımlılık eksik: '{package}'. Lütfen 'pip install -r requirements_enhanced.txt' komutunu çalıştırın.")
    return errors

def main():
    """Ana kontrol fonksiyonu."""
    print(f"{Style.BOLD}🚀 JetX Projesi Ön Kontrol Script'i Başlatılıyor...{Style.END}")
    print("-" * 60)

    all_errors = []
    all_warnings = []

    # 1. Yapılandırma Dosyası Kontrolü
    print(f"{Style.BLUE}1. Yapılandırma Dosyası Kontrolü (`config.yaml`){Style.END}")
    if YAML_ERROR:
        all_errors.append(YAML_ERROR)
        print_check("`config.yaml` dosyası ve ayarlar yüklendi.", False)
    else:
        config_exists = os.path.exists('config.yaml')
        print_check("`config.yaml` dosyası bulundu ve yüklendi.", config_exists)
        if not config_exists:
            all_errors.append("`config.yaml` dosyası proje kök dizininde bulunamadı.")
        else:
            hpo_errors = validate_hpo_space(CONFIG)
            profile_errors = validate_training_profiles(CONFIG)
            all_errors.extend(hpo_errors)
            all_errors.extend(profile_errors)
            print_check("HPO arama uzayı ve eğitim profilleri doğrulandı.", not (hpo_errors or profile_errors))

    # 2. Veri Uyumluluk Kontrolü
    print(f"\n{Style.BLUE}2. Veri ve Model Uyumluluk Kontrolü{Style.END}")
    data_errors = check_data_compatibility(CONFIG)
    all_errors.extend(data_errors)
    print_check("Veri sayısı, modellerin `sequence_length` değeri ile uyumlu.", not data_errors)

    # 3. Bağımlılık Kontrolü
    print(f"\n{Style.BLUE}3. Temel Bağımlılıkların Kontrolü{Style.END}")
    dep_errors = check_dependencies()
    all_errors.extend(dep_errors)
    print_check("Tüm temel kütüphaneler yüklü.", not dep_errors)

    # --- Sonuç Raporu ---
    print("\n" + "=" * 60)
    print(f"{Style.BOLD}📊 ÖN KONTROL SONUÇ RAPORU{Style.END}")
    print("=" * 60)

    if not all_errors and not all_warnings:
        print(f"{ICON_OK}{Style.GREEN}{Style.BOLD} Tüm kontroller başarıyla tamamlandı! Sistem eğitime hazır görünüyor.{Style.END}")
    else:
        if all_errors:
            print(f"{ICON_FAIL}{Style.RED}{Style.BOLD} Bulunan Hatalar ({len(all_errors)}):{Style.END}")
            for i, error in enumerate(all_errors, 1):
                print(f"  {i}. {error}")
        if all_warnings:
            print(f"\n{ICON_WARN}{Style.YELLOW}{Style.BOLD} Bulunan Uyarılar ({len(all_warnings)}):{Style.END}")
            for i, warning in enumerate(all_warnings, 1):
                print(f"  {i}. {warning}")

        print(f"\n{ICON_FAIL}{Style.RED}{Style.BOLD} Lütfen Colab'da eğitim başlatmadan önce yukarıdaki hataları düzeltin.{Style.END}")
        sys.exit(1)

if __name__ == "__main__":
    main()