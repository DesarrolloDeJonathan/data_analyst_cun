import sys
import os

def check_virtual_env():
    # Detecta si se está ejecutando dentro de un entorno virtual
    in_venv = (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )

    print("🔍 Estado del entorno virtual:")
    print("---------------------------------")
    print(f"Python ejecutándose desde: {sys.executable}")
    print(f"sys.prefix: {sys.prefix}")
    print(f"sys.base_prefix: {sys.base_prefix}")

    if in_venv:
        print("✅ Estás dentro de un entorno virtual.")
        # Busca si el entorno está en una carpeta estándar
        if os.path.exists(os.path.join(sys.prefix, 'pyvenv.cfg')):
            print(f"📁 Archivo de configuración encontrado en: {os.path.join(sys.prefix, 'pyvenv.cfg')}")
    else:
        print("❌ No estás dentro de un entorno virtual.")

if __name__ == "__main__":
    check_virtual_env()
