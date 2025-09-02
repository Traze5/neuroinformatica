# pages/00_Login.py — Login móvil (Open Gateway NV sandbox) → web login si no hay vínculo
import json, re, hashlib, requests, streamlit as st
from modules.ui import load_users  # para buscar MSISDN en usuarios.csv

st.set_page_config(page_title="Open Gateway - NV (Login)", page_icon="📱", layout="centered")

API_VERIFY = "https://sandbox.opengateway.telefonica.com/apigateway/number-verification/v0/verify"
MOCK_TOKEN = "mock_sandbox_access_token"  # token de sandbox/mock

# ------------------------------- País / prefijos -------------------------------
PAISES = [
    ("es", "+34",  "🇪🇸 España (+34)"),
    ("ec", "+593", "🇪🇨 Ecuador (+593)"),
    ("mx", "+52",  "🇲🇽 México (+52)"),
    ("us", "+1",   "🇺🇸 USA (+1)"),
]
LABEL_TO_PREFIX = {lbl: pref for _, pref, lbl in PAISES}

def _mask_msisdn(e164: str, country_prefix: str) -> str:
    local = e164.replace(country_prefix, "", 1)
    if len(local) <= 2:
        return f"{country_prefix} {local}"
    return f"{country_prefix} " + ("•" * (len(local)-2)) + local[-2:]

# ------------------------------- Estado persistente ----------------------------
for k, v in {
    "nv_ready": False,          # hay sesión lista para entrar con botón
    "nv_msisdn": "",            # e164 verificado
    "nv_country_prefix": "",    # prefijo del país
    "nv_bind_row": None,        # fila del CSV (dict)
}.items():
    st.session_state.setdefault(k, v)

# ------------------------------- Helpers NV -----------------------------------
def is_e164(s: str) -> bool:
    return bool(re.fullmatch(r"\+[1-9]\d{5,14}", (s or "").strip()))

def verify_number(msisdn: str, hashed: bool = False):
    headers = {
        "Authorization": f"Bearer {MOCK_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = (
        {"hashedPhoneNumber": hashlib.sha256(msisdn.encode()).hexdigest()}
        if hashed else
        {"phoneNumber": msisdn}
    )
    try:
        r = requests.post(API_VERIFY, headers=headers, json=payload, timeout=20)
        try:
            data = r.json()
        except Exception:
            data = {"_raw": r.text}
        return r.status_code, data
    except Exception as e:
        return None, {"error": str(e)}

def _match_msisdn_in_csv(e164: str):
    """
    Devuelve la fila del CSV si el MSISDN está vinculado.
    Acepta 'msisdn' en claro (+34...) o sha256(E.164).
    """
    df = load_users()
    if df.empty or "msisdn" not in df.columns:
        return None
    e164_clean = e164.strip()
    e164_hash  = hashlib.sha256(e164_clean.encode()).hexdigest()
    col = df["msisdn"].astype(str).str.strip()
    m = df[col.isin([e164_clean, e164_hash])]
    return None if m.empty else m.iloc[0].to_dict()

# ----------------------------------- UI ---------------------------------------
st.title("Number Verification (Open Gateway) — Login")
st.caption("Sandbox (mock) para prototipado. No requiere OIDC en esta demo.")

# Asegura CSV fresco cada intento
load_users.clear()

# Por defecto OFF para que veas la respuesta y aparezca el botón manual
auto_login = st.toggle("Entrar automáticamente si la verificación es exitosa", value=False)

cols = st.columns([2, 1])
with cols[0]:
    labels = [p[2] for p in PAISES]
    sel_label = st.selectbox("País / Código", labels, index=0)
    country_prefix = LABEL_TO_PREFIX[sel_label]

    local_number = st.text_input(
        "Número local (SIN código de país)",
        value=st.session_state.get("last_local", ""),
        help="Solo dígitos; no incluyas el prefijo del país."
    )
with cols[1]:
    use_hash = st.checkbox("Enviar hash (SHA-256)", help="Envía hashedPhoneNumber en vez de phoneNumber.")

# Construye E.164 a partir de prefijo + local
local_digits = re.sub(r"\D", "", local_number or "")
msisdn_e164  = f"{country_prefix}{local_digits}" if local_digits else ""

# --------------------------- Paso 1: Verificación NV ---------------------------
if st.button("Verificar ahora", type="primary", use_container_width=True, key="btn_verify"):
    # limpiamos cualquier intento previo
    st.session_state["nv_ready"] = False
    st.session_state["nv_bind_row"] = None

    if not is_e164(msisdn_e164):
        st.error("Formato inválido. Selecciona el país y escribe el número sin prefijo.")
        st.stop()

    st.session_state["last_local"] = local_number
    st.write("Llamando a:", f"`POST {API_VERIFY}`")
    status, data = verify_number(msisdn_e164, hashed=use_hash)

    if status is None:
        st.error(f"Error de red/cliente: {data.get('error')}")
    else:
        ok = data.get("devicePhoneNumberVerified", None)

        if status == 200 and ok is True:
            st.success("✅ Número verificado (sandbox).")

            # ¿Está vinculado en usuarios.csv?
            row = _match_msisdn_in_csv(msisdn_e164)
            if row is not None:
                # Guardamos estado para el paso 2 (fuera del if)
                st.session_state["nv_ready"] = True
                st.session_state["nv_msisdn"] = msisdn_e164
                st.session_state["nv_country_prefix"] = country_prefix
                st.session_state["nv_bind_row"] = row

                if auto_login:
                    # Autologin inmediato
                    st.session_state["auth_ok"] = True
                    st.session_state["usuario"] = row.get("usuario", "")
                    st.session_state["user_msisdn"] = msisdn_e164
                    st.switch_page("main.py")
            else:
                # Verificado pero NO vinculado → ir a login web para asociar
                st.info("Número verificado. Completa el login web para asociar tu usuario.")
                st.session_state["nv_ok"] = True
                st.switch_page("pages/00_Web_Login.py")

        elif status == 200 and ok is False:
            st.error("❌ Número NO verificado.")
        elif status == 401:
            st.error("401 Unauthorized — token mock inválido.")
        else:
            st.warning(f"HTTP {status} — revisa payload.")

        with st.expander("Respuesta completa"):
            st.code(json.dumps(data, indent=2, ensure_ascii=False), language="json")

# ---------------------- Paso 2: Botón persistente de entrada -------------------
if st.session_state.get("nv_ready") and not auto_login:
    row = st.session_state.get("nv_bind_row") or {}
    masked = _mask_msisdn(st.session_state["nv_msisdn"], st.session_state["nv_country_prefix"])
    st.success(f"Listo para entrar como: **{row.get('usuario','')}** · {masked}")

    if st.button("Ingresar al Simulador", type="primary", use_container_width=True, key="btn_go_main"):
        st.session_state["auth_ok"] = True
        st.session_state["usuario"] = row.get("usuario", "")
        st.session_state["user_msisdn"] = st.session_state["nv_msisdn"]
        # limpiar flags de NV
        st.session_state["nv_ready"] = False
        st.session_state["nv_bind_row"] = None
        st.switch_page("main.py")
