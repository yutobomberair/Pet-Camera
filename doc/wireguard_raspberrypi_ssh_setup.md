# WireGuardを使って家Aから家BのRaspberry PiへSSH接続する方法

## 概要
家Bに設置したRaspberry PiをWireGuard VPNサーバーとして構成し、  
家AのホストPCから安全にSSH接続する手順をまとめる。

---

## 構成

```text
[家A PC]
   ↓ (WireGuard VPN)
インターネット
   ↓
[家Bルーター（UDP 51820開放）]
   ↓
[Raspberry Pi（WireGuard Server）]
   ↓
[SSH接続]
```

---

## 前提条件

- 家Bに常時起動しているRaspberry Piがある
- 家Bのルーター設定を変更できる
- Raspberry PiにSSH接続済み（初期セットアップ完了済み）
- Raspberry Pi OS / Debian系OSを使用

---

## 使用技術

- VPN: WireGuard
- リモート接続: SSH

---

## 1. 家B（Raspberry Pi側）の設定

### 1-1. WireGuardインストール

```bash
sudo apt update
sudo apt install wireguard
```

### 1-2. 鍵生成

```bash
wg genkey | tee privatekey | wg pubkey > publickey
```

### 1-3. WireGuard設定ファイル作成

`/etc/wireguard/wg0.conf`

```ini
[Interface]
PrivateKey = <RaspberryPiの秘密鍵>
Address = 10.0.0.1/24
ListenPort = 51820
```

> ※ 今回はRaspberry Pi自身にSSHするだけなのでNAT設定不要

### 1-4. WireGuard起動

```bash
sudo wg-quick up wg0
sudo systemctl enable wg-quick@wg0
```

## 2. 家Bルーター設定

- UDP 51820 を Raspberry Pi にポートフォワード
- http://<家BのIPアドレス>を任意のブラウザで検索し、ポートフォワード設定を行う
- 調べ方はRaspi上で下記を実行し、default via ~のIPを引っ張ってくる
```bash
ip route
```

## 3. 家A（クライアントPC側）の設定

### 3-1. 鍵生成

```bash
wg genkey | tee privatekey | wg pubkey > publickey
```

### 3-2. クライアント設定ファイル

```ini
[Interface]
PrivateKey = <家A秘密鍵>
Address = 10.0.0.2/24

[Peer]
PublicKey = <RaspberryPi公開鍵>
Endpoint = <家BグローバルIP>:51820
AllowedIPs = 10.0.0.1/32
PersistentKeepalive = 25
```
- ここに指定するグローバルIPはRaspi上で下記を実行する
```bash
curl ifconfig.me
```
- またはブラウザで下記を検索
```bash
what is my ip
```

## 4. Raspberry Pi側にクライアント登録

```ini
[Peer]
PublicKey = <家A公開鍵>
AllowedIPs = 10.0.0.2/32
```

## 5. VPN接続

```bash
sudo wg-quick up wg0
```
```bash
[#] ip link add wg0 type wireguard
[#] wg setconf wg0 /dev/fd/63
[#] ip -4 address add 10.0.0.1/24 dev wg0
[#] ip link set mtu 1420 up dev wg0
```
※前に立ち上げたwg0がある場合は一度落とす必要がある
```bash
sudo wg-quick down wg0
```

## 6. SSH接続

```bash
ssh pi@10.0.0.1
```

## 7. 接続できない場合(今回私が該当)
家Bルーターの WAN IP が：

```text
100.xxx.xxx.xxx
```

だったため、CGNAT環境だった。

この場合：

- ポートフォワードしても外部から到達できない
- UDP 51820 が届かない
- handshake が発生しない

ため、Tailscale に切り替えた。


---

### CGNAT とは？

### CGNAT の概要

CGNAT（Carrier Grade NAT）は、ISP側で多数の利用者がグローバルIPを共有する仕組み。

構成イメージ：

```text
Internet
   ↓
ISP共有グローバルIP
   ↓
CGNAT
   ↓
自宅ルーター(WAN IP = 100.x.x.x)
   ↓
家庭内ネットワーク
```

---

### なぜ問題になるのか？

通常のポートフォワードは：

```text
Internet
   ↓
グローバルIP
   ↓
自宅ルーター
   ↓
Raspberry Pi
```

という前提で動作する。

しかし CGNAT 環境では：

- ISP側にもNATが存在する
- 自宅ルーターが本当のグローバルIPを持っていない
- 外部から直接着信できない

ため、自宅ルーターでポート開放しても通信が届かない。

---

### CGNAT の特徴

#### WAN IP が以下で始まることが多い

```text
100.x.x.x
```

または：

```text
10.x.x.x
```

---

### 今回の症状

今回発生していた：

- WireGuard handshake が発生しない
- transfer sent のみ増える
- received が 0 B
- tcpdump に UDP 51820 が届かない

という症状は、CGNAT 環境と一致していた。

---

### CGNAT 環境での対策

#### 方法1：Tailscale を使う（今回採用）

メリット：

- ポート開放不要
- CGNAT対応
- WireGuardベース
- NAT越え自動

---

#### 方法2：固定グローバルIP契約

ISPによってはオプション契約で：

- 固定IP
- ポート開放可能IPv4

を利用できる場合がある。

---

#### 方法3：VPS中継

VPSをWireGuardサーバにして中継する構成。

ただし構築難易度は高め。
