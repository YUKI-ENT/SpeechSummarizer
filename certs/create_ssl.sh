#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CERT_DIR="${CERT_DIR:-$SCRIPT_DIR}"
C="${C:-JP}"
ST="${ST:-Tokyo}"
L="${L:-Chiyoda}"
O="${O:-SpeechSummarizer}"
OU="${OU:-Development}"
HOSTNAME="${HOSTNAME:-localhost}"
DNS_ALT="${DNS_ALT:-}"
IP_ALT="${IP_ALT:-127.0.0.1}"
CA_CN="${CA_CN:-SpeechSummarizer Local CA}"
CA_DAYS="${CA_DAYS:-3650}"
SERVER_DAYS="${SERVER_DAYS:-825}"
CA_KEY_NAME="${CA_KEY_NAME:-myCA.key}"
CA_CERT_NAME="${CA_CERT_NAME:-myCA.pem}"
SERVER_KEY_NAME="${SERVER_KEY_NAME:-key.pem}"
SERVER_CERT_NAME="${SERVER_CERT_NAME:-cert.pem}"
SERVER_CSR_NAME="${SERVER_CSR_NAME:-server.csr}"
SERIAL_NAME="${SERIAL_NAME:-myCA.srl}"

usage() {
  cat <<'EOF'
ローカル HTTPS 用の自己署名 CA とサーバー証明書を作成します。

使い方:
  ./create_ssl.sh
  ./create_ssl.sh --hostname my-pc --dns-alt my-pc.local --ip-alt 192.168.1.10
  ./create_ssl.sh --cert-dir ./certs --org "Clinic Demo" --unit "IT"

主なオプション:
  --hostname NAME        サーバー証明書の CN / SAN に入れるホスト名
  --dns-alt NAME         追加の DNS 名。複数指定可
  --ip-alt ADDRESS       追加の IP アドレス。複数指定可
  --cert-dir PATH        出力先ディレクトリ
  --country CODE         Subject の C
  --state TEXT           Subject の ST
  --locality TEXT        Subject の L
  --org TEXT             Subject の O
  --unit TEXT            Subject の OU
  --ca-common-name TEXT  CA 証明書の CN
  --ca-days DAYS         CA 証明書の有効日数
  --server-days DAYS     サーバー証明書の有効日数
  --help                 このヘルプを表示

環境変数でも同じ値を上書きできます:
  HOSTNAME, DNS_ALT, IP_ALT, CERT_DIR, C, ST, L, O, OU, CA_CN

出力ファイル既定値:
  cert.pem   - config.json の ssl.certfile 用
  key.pem    - config.json の ssl.keyfile 用
  myCA.pem   - クライアントへ配布して信頼済みルートとして登録する CA 証明書

例:
  ./create_ssl.sh --hostname localhost
  ./create_ssl.sh --hostname speechsummarizer --dns-alt speechsummarizer.local --ip-alt 192.168.0.20
EOF
}

ensure_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "エラー: '$1' が見つかりません。OpenSSL をインストールしてから実行してください。" >&2
    exit 1
  fi
}

append_csv() {
  local current="$1"
  local value="$2"
  if [[ -z "$current" ]]; then
    printf '%s' "$value"
  else
    printf '%s,%s' "$current" "$value"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hostname)
      HOSTNAME="$2"
      shift 2
      ;;
    --dns-alt)
      DNS_ALT="$(append_csv "$DNS_ALT" "$2")"
      shift 2
      ;;
    --ip-alt)
      IP_ALT="$(append_csv "$IP_ALT" "$2")"
      shift 2
      ;;
    --cert-dir)
      CERT_DIR="$2"
      shift 2
      ;;
    --country)
      C="$2"
      shift 2
      ;;
    --state)
      ST="$2"
      shift 2
      ;;
    --locality)
      L="$2"
      shift 2
      ;;
    --org)
      O="$2"
      shift 2
      ;;
    --unit)
      OU="$2"
      shift 2
      ;;
    --ca-common-name)
      CA_CN="$2"
      shift 2
      ;;
    --ca-days)
      CA_DAYS="$2"
      shift 2
      ;;
    --server-days)
      SERVER_DAYS="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "不明な引数: $1" >&2
      echo >&2
      usage >&2
      exit 1
      ;;
  esac
done

ensure_command openssl

mkdir -p "$CERT_DIR"
CERT_DIR="$(cd "$CERT_DIR" && pwd)"

CA_KEY="$CERT_DIR/$CA_KEY_NAME"
CA_CERT="$CERT_DIR/$CA_CERT_NAME"
SERVER_KEY="$CERT_DIR/$SERVER_KEY_NAME"
SERVER_CSR="$CERT_DIR/$SERVER_CSR_NAME"
SERVER_CERT="$CERT_DIR/$SERVER_CERT_NAME"
CA_SERIAL="$CERT_DIR/$SERIAL_NAME"
EXT_FILE="$CERT_DIR/.server_cert_ext.cnf"

if [[ "$CERT_DIR" == "$SCRIPT_DIR" ]]; then
  CONFIG_CERT_PATH="certs/$SERVER_CERT_NAME"
  CONFIG_KEY_PATH="certs/$SERVER_KEY_NAME"
else
  CONFIG_CERT_PATH="$SERVER_CERT"
  CONFIG_KEY_PATH="$SERVER_KEY"
fi

SAN_LIST="DNS:${HOSTNAME}"

IFS=',' read -r -a dns_items <<< "$DNS_ALT"
for dns_name in "${dns_items[@]}"; do
  [[ -z "$dns_name" ]] && continue
  SAN_LIST="${SAN_LIST},DNS:${dns_name}"
done

IFS=',' read -r -a ip_items <<< "$IP_ALT"
for ip_addr in "${ip_items[@]}"; do
  [[ -z "$ip_addr" ]] && continue
  SAN_LIST="${SAN_LIST},IP:${ip_addr}"
done

cat > "$EXT_FILE" <<EOF
subjectAltName=${SAN_LIST}
keyUsage=digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
EOF

cleanup() {
  rm -f "$EXT_FILE"
}
trap cleanup EXIT

echo "== create_ssl.sh =="
echo "出力先      : $CERT_DIR"
echo "Subject     : /C=$C/ST=$ST/L=$L/O=$O/OU=$OU"
echo "ホスト名     : $HOSTNAME"
echo "SAN         : $SAN_LIST"
echo

echo "1. CA 証明書を確認します"
if [[ ! -f "$CA_CERT" || ! -f "$CA_KEY" ]]; then
  echo "   - CA を新規作成します"
  openssl genrsa -out "$CA_KEY" 4096
  openssl req -x509 -new -nodes -key "$CA_KEY" -sha256 -days "$CA_DAYS" \
    -subj "/C=$C/ST=$ST/L=$L/O=$O/OU=$OU/CN=$CA_CN" \
    -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -out "$CA_CERT"
else
  echo "   - 既存の CA を再利用します"
fi

echo
echo "2. サーバー秘密鍵と CSR を作成します"
openssl genrsa -out "$SERVER_KEY" 4096
openssl req -new -key "$SERVER_KEY" \
  -subj "/C=$C/ST=$ST/L=$L/O=$O/OU=$OU/CN=$HOSTNAME" \
  -addext "subjectAltName=$SAN_LIST" \
  -out "$SERVER_CSR"

echo
echo "3. サーバー証明書に署名します"
openssl x509 -req -in "$SERVER_CSR" \
  -CA "$CA_CERT" -CAkey "$CA_KEY" -CAserial "$CA_SERIAL" -CAcreateserial \
  -out "$SERVER_CERT" -days "$SERVER_DAYS" -sha256 \
  -extfile "$EXT_FILE"

echo
echo "4. 生成結果を確認します"
openssl x509 -in "$SERVER_CERT" -noout -subject -issuer -ext subjectAltName

echo
echo "完了しました。"
echo
echo "[SpeechSummarizer の設定]"
echo "config.json の ssl は次のように設定できます。"
cat <<EOF
  "ssl": {
    "enabled": true,
    "certfile": "${CONFIG_CERT_PATH}",
    "keyfile": "${CONFIG_KEY_PATH}"
  }
EOF
echo
echo "注意:"
echo "  - ${SERVER_KEY} は秘密鍵です。配布しないでください。"
echo "  - クライアントへ配布するのは ${CA_CERT} です。"
echo
echo "[ブラウザへ証明書を入れる手順]"
echo "1. ${CA_CERT} をクライアント PC にコピーします。"
echo "2. Windows では ${CA_CERT} をダブルクリックして [証明書のインストール] を選びます。"
echo "3. [ローカル コンピューター] を選び、保存先は [信頼されたルート証明機関] を指定します。"
echo "4. インポート完了後、Chrome / Edge はいったん完全に終了してから再起動します。"
echo "5. 次の URL にアクセスして、証明書エラーが消えることを確認します。"
echo "   https://${HOSTNAME}:8000/"

for dns_name in "${dns_items[@]}"; do
  [[ -z "$dns_name" ]] && continue
  echo "   https://${dns_name}:8000/"
done

for ip_addr in "${ip_items[@]}"; do
  [[ -z "$ip_addr" ]] && continue
  echo "   https://${ip_addr}:8000/"
done

echo
echo "[Firefox を使う場合]"
echo "Firefox は独自の証明書ストアを使う場合があります。"
echo "  1. Firefox の [設定] -> [プライバシーとセキュリティ] -> [証明書を表示] を開きます。"
echo "  2. [認証局証明書] で ${CA_CERT} をインポートします。"
echo "  3. [この認証局によるウェブサイトの識別を信頼する] を有効にします。"
