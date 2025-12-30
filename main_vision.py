from security import barrera_seguridad
from modo_fotos import modo_fotos

if __name__ == "__main__":
    ok, pwd = barrera_seguridad(password=None)
    if ok:
        modo_fotos()
