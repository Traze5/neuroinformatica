# nv_demo.py
import json
import re
import hashlib
import requests
import streamlit as st

st.set_page_config(page_title="Open Gateway - Number Verification (Demo)", page_icon="✅", layout="centered")

API_VERIFY = "https://sandbox.opengateway.telefonica.com/apigateway/number-verification/v0/verify"
MOCK_TOKEN = "mock_sandbox_access_token"  # token de conveniencia del sandbox (mock)

# ---------- Helpers ----------
def is_e164(s: str) -> bool:
    # E.164: máximo 15 dígitos después del '+'
    return bool(re.fullmatch(r"\+[1-9]\d{5,14}", (s or "").strip()))

def verify_number(msisdn: str = None, hashed: bool = False):
    headers = {
        "Authorization": f"Bearer {MOCK_TOKEN}",
        "Content-Type": "application/json",
    }
    if hashed:
        payload = {"hashedPhoneNumber": hashlib.sha256(msisdn.encode()).hexdigest()}
    else:
        payload = {"phoneNumber": msisdn}

    try:
        r = requests.post(API_VERIFY, headers=headers, json=payload, timeout=20)
        status = r.status_code
        try:
            data = r.json()
        except Exception:
            data = {"_raw": r.text}
        return status, data
    except Exception as e:
        return None, {"error": str(e)}

# ---------- UI ----------
st.title("Number Verification (Open Gateway) — Demo rápido")
st.caption("Demostración con entorno sandbox (mock). No requiere OIDC ni secretos.")

colL, colR = st.columns([2,1])
with colL:
    msisdn = st.text_input("Número en formato E.164", value="+593991234561", help="Ej.: +593991234561 (Ecuador). Debe iniciar con '+'.")
with colR:
    use_hash = st.checkbox("Usar hashedPhoneNumber", help="Envía SHA-256(E.164) en lugar del número en claro.")

run = st.button("Verificar ahora", type="primary")

if run:
    if not is_e164(msisdn):
        st.error("Formato inválido. Usa E.164 (p. ej., +593991234561).")
        st.stop()
    st.write("Llamando a:", f"`POST {API_VERIFY}`")
    status, data = verify_number(msisdn, hashed=use_hash)

    if status is None:
        st.error(f"Error de red/cliente: {data.get('error')}")
    else:
        ok = data.get("devicePhoneNumberVerified")
        sandbox_note = data.get("_sandbox")
        if status == 200 and ok is True:
            st.success("✅ Número verificado (coincide).")
        elif status == 200 and ok is False:
            st.error("❌ Número NO verificado (no coincide).")
        elif status == 200:
            st.warning("⚠️ 200 OK, pero sin 'devicePhoneNumberVerified'. Revisa el payload.")
        elif status == 401:
            st.error("401 Unauthorized — revisa el Bearer token (mock).")
        else:
            st.warning(f"Respuesta HTTP {status}")

        if sandbox_note:
            st.info(sandbox_note)

        st.markdown("**Respuesta completa**")
        st.code(json.dumps(data, indent=2, ensure_ascii=False), language="json")

st.divider()
with st.expander("¿Cómo explicar esto en el TFM? (resumen breve)"):
    st.markdown(
        """
**Qué valida**: *posesión de línea*. El operador indica si el MSISDN enviado coincide con la línea asociada al token del dispositivo.

**Ventajas**:
- Menos fricción vs. OTP por SMS (no hay códigos que copiar).
- Menor superficie de ataque que SMS-OTP.
- Privacidad: puedes enviar `hashedPhoneNumber` (SHA-256 de E.164).

**Limitaciones**:
- En “real” requiere conexión por **datos móviles** (no Wi-Fi).
- No sustituye KYC: prueba posesión de SIM, no identidad legal.

**Integración real**:
- Este demo usa **mock** (Bearer fijo) para prototipado.
- En producción, obtén `access_token` vía **OIDC Authorization Code** y llama al mismo endpoint `POST /verify`.
        """
    )
