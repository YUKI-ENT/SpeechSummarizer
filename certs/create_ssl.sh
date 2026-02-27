#!/bin/bash
set -euo pipefail

# =========================================================
# create_ssl.sh  (distribution-friendly)
# - Create/Reuse a local CA (myCA.pem / myCA.key)
# - Issue a server certificate for uvicorn (ai-server.crt / ai-server.key)
#
# Usage examples:
#   ./create_ssl.sh
#   ST=Osaka L=Ibaraki O=YukiENTclinic OU=Hospital HOSTNAME=ai-server ./create_ssl.sh
#   HOSTNAME=ai-server DNS_ALT="ai-server.clinic.lan" IP_ALT="192.168.200.12" ./create_ssl.sh
#
# Outputs (default):
#   certs/myCA.pem              -> import into client "Trusted Root CA"
#   certs/ai-server.crt         -> set to ssl_certfile in config.json
#   certs/ai-server.key         -> set to ssl_keyfile  in config.json (KEEP SECRET)
# =========================================================

# ---- Where to put certs (relative to project root) ----
CERT_DIR="."

# ---- Certificate subject fields (customize via env vars) ----
C="${C:-JP}"
ST="${ST:-Osaka}"
L="${L:-Ibaraki}"
O="${O:-YukiENTclinic}"
OU="${OU:-Hospital}"

# ---- Server name (the URL host you access) ----
HOSTNAME="${HOSTNAME:-ai-server}"

# Optional additional SAN entries
# - DNS_ALT: extra DNS name (e.g. ai-server.clinic.lan)
# - IP_ALT : IP address if you also access by IP (e.g. 192.168.200.12)
DNS_ALT="${DNS_ALT:-}"
IP_ALT="${IP_ALT:-}"

# ---- CA identity (CN is shown on clients) ----
CA_CN="${CA_CN:-${O} Local CA}"

# ---- File names ----
CA_KEY="$CERT_DIR/myCA.key"
CA_CERT="$CERT_DIR/myCA.pem"
SERVER_KEY="$CERT_DIR/${HOSTNAME}.key"
SERVER_CSR="$CERT_DIR/${HOSTNAME}.csr"
SERVER_CERT="$CERT_DIR/${HOSTNAME}.crt"
CA_SERIAL="$CERT_DIR/myCA.srl"

mkdir -p "$CERT_DIR"
cd "$CERT_DIR"

# Build SAN string
SAN_LIST="DNS:${HOSTNAME}"
if [[ -n "$DNS_ALT" ]]; then
  SAN_LIST="${SAN_LIST},DNS:${DNS_ALT}"
fi
if [[ -n "$IP_ALT" ]]; then
  SAN_LIST="${SAN_LIST},IP:${IP_ALT}"
fi

echo "== create_ssl.sh =="
echo "CERT_DIR : $CERT_DIR"
echo "SUBJECT  : /C=$C/ST=$ST/L=$L/O=$O/OU=$OU"
echo "HOSTNAME : $HOSTNAME"
echo "SAN      : $SAN_LIST"
echo

echo "=== 1) Create CA (if not exists) ==="
if [[ ! -f "$CA_CERT" || ! -f "$CA_KEY" ]]; then
  echo "-> generating CA key/cert..."
  openssl genrsa -out "$CA_KEY" 4096

  openssl req -x509 -new -nodes -key "$CA_KEY" -sha256 -days 3650 \
    -subj "/C=$C/ST=$ST/L=$L/O=$O/OU=$OU/CN=$CA_CN" \
    -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -out "$CA_CERT"

  echo "✔ CA created: $CA_CERT"
else
  echo "✔ CA exists : $CA_CERT"
fi

echo
echo "=== 2) Create server key + CSR (SAN included) ==="
echo "-> generating server private key..."
openssl genrsa -out "$SERVER_KEY" 4096

echo "-> generating CSR..."
openssl req -new -key "$SERVER_KEY" \
  -subj "/C=$C/ST=$ST/L=$L/O=$O/OU=$OU/CN=$HOSTNAME" \
  -addext "subjectAltName=$SAN_LIST" \
  -out "$SERVER_CSR"

echo
echo "=== 3) Sign server certificate with CA ==="
# Note: -CAcreateserial will create myCA.srl in current dir; we keep it under CERT_DIR.
# If OpenSSL writes it elsewhere, we move it back.
openssl x509 -req -in "$SERVER_CSR" \
  -CA "$CA_CERT" -CAkey "$CA_KEY" -CAcreateserial \
  -out "$SERVER_CERT" -days 825 -sha256 \
  -extfile <(printf "subjectAltName=%s\nkeyUsage=digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n" "$SAN_LIST")

# Keep serial file under CERT_DIR (openssl uses <CAcert>.srl or myCA.srl depending)
if [[ -f "./myCA.srl" && ! -f "$CA_SERIAL" ]]; then
  mv -f "./myCA.srl" "$CA_SERIAL"
fi
if [[ -f "./${CA_CERT}.srl" ]]; then
  mv -f "./${CA_CERT}.srl" "$CA_SERIAL" || true
fi

echo
echo "=== 4) Verify outputs ==="
openssl x509 -in "$SERVER_CERT" -noout -subject -issuer -ext subjectAltName

echo
echo "========================================================="
echo "✔ Done."
echo
echo "[Server side / config.json]"
echo "  ssl_certfile : $SERVER_CERT"
echo "  ssl_keyfile  : $SERVER_KEY"
echo "  ※ server key ($SERVER_KEY) is SECRET. Do NOT distribute it."
echo
echo "[Client side (Windows/Edge/Chrome)]"
echo "  Import this file as 'Trusted Root Certification Authorities':"
echo "    $CA_CERT"
echo "  After import, restart the browser and open:"
echo "    https://${HOSTNAME}:8000/"
if [[ -n "$DNS_ALT" ]]; then
  echo "    https://${DNS_ALT}:8000/"
fi
if [[ -n "$IP_ALT" ]]; then
  echo "  If you access by IP, this is also covered:"
  echo "    https://${IP_ALT}:8000/"
fi
echo "========================================================="
