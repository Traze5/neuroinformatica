# pages/00_Login.py — Login móvil (Open Gateway NV sandbox) → redirección a login web
import json, re, hashlib, requests, streamlit as st

st.set_page_config(page_title="Open Gateway - NV (Login)", page_icon="✅", layout="centered")

API_VERIFY = "https://sandbox.opengateway.telefonica.com/apigateway/number-verification/v0/verify"
MOCK_TOKEN = "mock_sandbox_access_token"  # token de sandbox/mock

# -------------------------------------------------------------------
# Estado persistente para el 2º paso (botón "Entrar al panel")
# -------------------------------------------------------------------
if "nv_success" not in st.session_state:
    st.session_state["nv_success"] = False
if "user_msisdn" not in st.session_state:
    st.session_state["user_msisdn"] = ""

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def is_e164(s: str) -> bool:
    # E.164: '+' + 6..15 dígitos (primero 1-9)
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

def goto_main():
    # Marcamos NV OK y saltamos al login web para vincular usuario → luego al main
    st.session_state["nv_ok"] = True
    st.switch_page("pages/00_Web_Login.py")

# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
st.title("Number Verification (Open Gateway) — Login")
st.caption("Sandbox (mock) para prototipado. No requiere OIDC en esta demo.")

auto_login = st.toggle("Entrar automáticamente si la verificación es exitosa", value=False)

colL, colR = st.columns([2, 1])
with colL:
    msisdn = st.text_input(
        "Número en formato E.164",
        value=st.session_state.get("last_msisdn", "+593984948789"),
        help="Ej.: +593991234561 (debe iniciar con '+')."
    )
with colR:
    use_hash = st.checkbox("Enviar hash (SHA-256)", help="Usa hashedPhoneNumber en vez de phoneNumber.")

# Paso 1: verificar ahora
if st.button("Verificar ahora", type="primary", use_container_width=True, key="btn_verify"):
    if not is_e164(msisdn):
        st.error("Formato inválido. Usa E.164 (p. ej., +593991234561).")
        st.stop()

    st.session_state["last_msisdn"] = msisdn
    st.write("Llamando a:", f"`POST {API_VERIFY}`")
    status, data = verify_number(msisdn, hashed=use_hash)

    if status is None:
        st.error(f"Error de red/cliente: {data.get('error')}")
    else:
        ok = data.get("devicePhoneNumberVerified", None)

        if status == 200 and ok is True:
            st.success("✅ Número verificado (sandbox).")
            # Guardamos msisdn (en claro o como etiqueta informativa si usaste hash)
            st.session_state["user_msisdn"] = msisdn if not use_hash else f"sha256({msisdn})"
            st.session_state["nv_success"] = True  # habilita el 2º paso

            if auto_login:
                goto_main()  # salto inmediato
        elif status == 200 and ok is False:
            st.session_state["nv_success"] = False
            st.error("❌ Número NO verificado.")
        elif status == 401:
            st.session_state["nv_success"] = False
            st.error("401 Unauthorized — token mock inválido.")
        else:
            st.session_state["nv_success"] = False
            st.warning(f"HTTP {status} — revisa payload.")

        with st.expander("Respuesta completa"):
            st.code(json.dumps(data, indent=2, ensure_ascii=False), language="json")

st.divider()

# Paso 2: botón "Entrar al panel" PERSISTENTE (fuera del if de verificación)
if st.session_state.get("nv_success") and not auto_login:
    st.success(f"MSISDN verificado: {st.session_state.get('user_msisdn')}")
    if st.button("Ingresar al Simulador", type="primary", use_container_width=True, key="btn_go_panel"):
        goto_main()

with st.expander("Notas (TFM)"):
    st.markdown(
        "- Prueba de **posesión de línea** (SIM) sin OTP por SMS.\n"
        "- Menos fricción y ataque que SMS-OTP.\n"
        "- En prod: OIDC → `access_token` → `POST /verify`.\n"
        "- Este login usa sandbox/mock del operador."
    )
