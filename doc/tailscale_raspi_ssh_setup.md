# Raspberry Pi に Tailscale 経由で SSH 接続する手順

## 概要

家A（WSL環境）から、家Bにある Raspberry Pi へ SSH 接続する。

今回は WireGuard を直接公開しようとしたが、家B回線が CGNAT 環境だったため、Tailscale を使用する構成に変更した。

---

# 構成

```text
家A (WSL)
    ↓
Tailscale VPN
    ↓
家B Raspberry Pi
```

---

# 1. Raspberry Pi 側設定

## 1-1. Tailscale インストール

```bash
curl -fsSL https://tailscale.com/install.sh | sh
```

---

## 1-2. Tailscale 起動

```bash
sudo tailscale up
```

実行すると認証URLが表示される。

例：

```text
https://login.tailscale.com/a/xxxxxxxx
```

---

## 1-3. ブラウザで認証

表示されたURLをブラウザで開き、ログインして「Connect」する。

※ Raspberry Pi 自身にGUIは不要  
※ 別PCのブラウザで開いてOK

---

## 1-4. 接続確認

```bash
tailscale status
```

または：

```bash
tailscale ip
```

例：

```text
100.xxx.xxx.xxx
```

---

# 2. 家A（WSL）側設定

## 2-1. Tailscale インストール

```bash
curl -fsSL https://tailscale.com/install.sh | sh
```

---

## 2-2. Tailscale 起動

```bash
sudo tailscale up
```

表示されたURLをブラウザで開き、同じアカウントでログイン＆Connect。

---

## 2-3. 接続確認

```bash
tailscale status
```

Raspberry Pi が表示されればOK。

---

# 3. SSH接続

## Raspberry Pi の Tailscale IP確認

Raspberry Pi側：

```bash
tailscale ip
```

例：

```text
100.xxx.xxx.xxx
```

---

## WSLからSSH

```bash
ssh <ユーザー名>@100.xxx.xxx.xxx
```

例：

```bash
ssh oonumayuto@100.xxx.xxx.xxx
```

---

# 4. 補足

## Tailscale のメリット

- ポート開放不要
- CGNAT対応
- WireGuardベース
- グローバルIP不要
- 外出先からでも接続可能

---

## 自動起動

通常、Tailscale は自動起動されるため、一度設定すれば基本的に常時接続される。

---

## 状態確認

```bash
tailscale status
```

---

## サービス確認（Raspberry Pi）

```bash
systemctl status tailscaled
```

---

# 5. 今回詰まったポイント

## WireGuard直接公開が失敗した原因

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
