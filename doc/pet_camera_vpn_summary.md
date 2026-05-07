# Pet Camera リモート監視構成メモ

## 概要
Pet Camera システムで、犬検知時にスマホへ通知し、必要時のみストリーミング視聴できる構成。

## 想定フロー
1. 犬検知
2. Discord 通知送信
3. スマホで通知確認
4. VPN (WireGuard) 経由で自宅ネットワークへ接続
5. ストリームURLへアクセス
6. Raspberry Pi 側でアクセス検知してストリーミング開始

## WireGuard VPN の考え方
- VPN参加 = 外部端末を自宅LANの一員として扱えるようにすること

## 仮想IP例
- Raspberry Pi: 10.0.0.1
- 開発PC: 10.0.0.2
- スマホ: 10.0.0.3

## Raspberry Pi 側設定例
```ini
[Interface]
PrivateKey = <RasPi秘密鍵>
Address = 10.0.0.1/24

[Peer]
PublicKey = <スマホ公開鍵>
AllowedIPs = 10.0.0.3/32
```

## スマホ側設定例
```ini
[Interface]
PrivateKey = <スマホ秘密鍵>
Address = 10.0.0.3/24

[Peer]
PublicKey = <RasPi公開鍵>
Endpoint = <グローバルIP>:51820
AllowedIPs = 10.0.0.0/24
PersistentKeepalive = 25
```

## Discord通知イメージ
🐶 犬を検知しました  
ライブを見る: http://10.0.0.1:8000/stream

## ストリーミング構成案
- Raspberry Pi で HTTP サーバを起動
- `/stream` で配信
- アクセス検知時のみストリーミングON可能

## 実装ステップ
1. WireGuardでスマホ接続確認
2. RasPi疎通確認
3. HTTPサーバ確認
4. ストリーム配信実装
5. Discord通知連携
