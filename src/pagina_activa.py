import subprocess
import time
import sys


CACHE = {
    "t": 0.0,
    "data": {
        "active_app": "",
        "window_title": "",
        "browser": "",
        "page_title": "",
        "url": "",
        "mouse_x": "",
        "mouse_y": "",
        "active_screen": "",
        "screen_count": "",
    }
}


def _osascript(script, timeout=3.5):
    try:
        resultado = subprocess.run(
            ["osascript"],
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if resultado.returncode != 0:
            return ""
        return resultado.stdout.strip()
    except Exception:
        return ""


def _limpiar(texto):
    if texto is None:
        return ""
    return str(texto).replace("\r", " ").replace("\n", " ").strip()


def _apple_text(texto):
    texto = str(texto).replace("\\", "\\\\").replace('"', '\\"')
    return '"' + texto + '"'


def mouse_y_pantalla():
    datos = {
        "mouse_x": "",
        "mouse_y": "",
        "active_screen": "",
        "screen_count": "",
    }

    try:
        from AppKit import NSEvent, NSScreen

        punto = NSEvent.mouseLocation()
        x = float(punto.x)
        y = float(punto.y)

        pantallas = NSScreen.screens()
        datos["mouse_x"] = round(x, 2)
        datos["mouse_y"] = round(y, 2)
        datos["screen_count"] = len(pantallas)

        for i, pantalla in enumerate(pantallas):
            frame = pantalla.frame()
            try:
                nombre = str(pantalla.localizedName())
            except Exception:
                nombre = "screen"

            dentro_x = frame.origin.x <= x <= frame.origin.x + frame.size.width
            dentro_y = frame.origin.y <= y <= frame.origin.y + frame.size.height

            if dentro_x and dentro_y:
                datos["active_screen"] = f"{i}:{nombre}"
                break

        if datos["active_screen"] == "" and pantallas:
            datos["active_screen"] = "unknown"

    except Exception:
        pass

    return datos


def app_y_ventana_activa():
    script = '''
    tell application "System Events"
        try
            set frontApp to first application process whose frontmost is true
            set appName to name of frontApp
            set winName to ""
            try
                set winName to name of front window of frontApp
            end try
            return appName & linefeed & winName
        on error
            return ""
        end try
    end tell
    '''
    salida = _osascript(script)
    partes = salida.splitlines()
    app = partes[0] if len(partes) > 0 else ""
    ventana = partes[1] if len(partes) > 1 else ""
    return _limpiar(app), _limpiar(ventana)


def datos_chrome_like(nombre_app, titulo_ventana=""):
    objetivo = _apple_text(titulo_ventana)

    script = f'''
    tell application "{nombre_app}"
        try
            if (count of windows) = 0 then return ""

            set objetivo to {objetivo}
            set mejorTitulo to ""
            set mejorURL to ""

            repeat with w in windows
                try
                    repeat with t in tabs of w
                        set tituloPagina to title of t
                        set urlPagina to URL of t

                        if tituloPagina is objetivo then
                            return tituloPagina & linefeed & urlPagina
                        end if

                        if objetivo is not "" then
                            if objetivo contains tituloPagina then
                                set mejorTitulo to tituloPagina
                                set mejorURL to urlPagina
                            else if tituloPagina contains objetivo then
                                set mejorTitulo to tituloPagina
                                set mejorURL to urlPagina
                            end if
                        end if
                    end repeat
                end try
            end repeat

            if mejorURL is not "" then
                return mejorTitulo & linefeed & mejorURL
            end if

            try
                set tituloPagina to title of active tab of front window
                set urlPagina to URL of active tab of front window
                return tituloPagina & linefeed & urlPagina
            on error
                return ""
            end try
        on error
            return ""
        end try
    end tell
    '''

    salida = _osascript(script)
    partes = salida.splitlines()
    titulo = partes[0] if len(partes) > 0 else ""
    url = partes[1] if len(partes) > 1 else ""
    return _limpiar(titulo), _limpiar(url)


def datos_safari(titulo_ventana=""):
    objetivo = _apple_text(titulo_ventana)

    script = f'''
    tell application "Safari"
        try
            if (count of windows) = 0 then return ""

            set objetivo to {objetivo}
            set mejorTitulo to ""
            set mejorURL to ""

            repeat with w in windows
                try
                    repeat with t in tabs of w
                        set tituloPagina to name of t
                        set urlPagina to URL of t

                        if tituloPagina is objetivo then
                            return tituloPagina & linefeed & urlPagina
                        end if

                        if objetivo is not "" then
                            if objetivo contains tituloPagina then
                                set mejorTitulo to tituloPagina
                                set mejorURL to urlPagina
                            else if tituloPagina contains objetivo then
                                set mejorTitulo to tituloPagina
                                set mejorURL to urlPagina
                            end if
                        end if
                    end repeat
                end try
            end repeat

            if mejorURL is not "" then
                return mejorTitulo & linefeed & mejorURL
            end if

            try
                set tituloPagina to name of current tab of front window
                set urlPagina to URL of current tab of front window
                return tituloPagina & linefeed & urlPagina
            on error
                return ""
            end try
        on error
            return ""
        end try
    end tell
    '''

    salida = _osascript(script)
    partes = salida.splitlines()
    titulo = partes[0] if len(partes) > 0 else ""
    url = partes[1] if len(partes) > 1 else ""
    return _limpiar(titulo), _limpiar(url)


def obtener_pagina_activa(forzar=False):
    ahora = time.time()

    if not forzar and ahora - CACHE["t"] < 0.75:
        return CACHE["data"]

    app, titulo_ventana = app_y_ventana_activa()
    mouse = mouse_y_pantalla()

    navegadores_chromium = {
        "Google Chrome",
        "Google Chrome Canary",
        "Microsoft Edge",
        "Brave Browser",
        "Arc",
        "Chromium",
        "Opera",
        "Vivaldi",
    }

    browser = ""
    page_title = ""
    url = ""

    if app in navegadores_chromium:
        browser = app
        page_title, url = datos_chrome_like(app, titulo_ventana)
    elif app in {"Safari", "Safari Technology Preview"}:
        browser = app
        page_title, url = datos_safari(titulo_ventana)

    data = {
        "active_app": app,
        "window_title": titulo_ventana,
        "browser": browser,
        "page_title": page_title,
        "url": url,
        "mouse_x": mouse.get("mouse_x", ""),
        "mouse_y": mouse.get("mouse_y", ""),
        "active_screen": mouse.get("active_screen", ""),
        "screen_count": mouse.get("screen_count", ""),
    }

    CACHE["t"] = ahora
    CACHE["data"] = data
    return data


if __name__ == "__main__":
    if "--delay" in sys.argv:
        print("Da click en la ventana o pantalla que quieres registrar. Leyendo en 3 segundos...")
        time.sleep(3)
    print(obtener_pagina_activa(forzar=True))
