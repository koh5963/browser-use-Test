# Browser-Use-Test

Sandbox project for experimenting with **browser-use** automation and Vision-based UI analysis.
It documents how to install dependencies, configure API keys, and run a small sample task.

* Free software: MIT License

## Features

* browser-use を用いた各種自動化検証
* Vision 機能の挙動確認
* 将来的には仕様書を読み込んでテストケース自動生成→自動テスト→エビデンス作成まで

## Requirements

* Python 3.9+
* OpenAI API キー
* Playwright で使用するブラウザ（`playwright install` でセットアップ）

## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.

## Installation

```bash
python -m pip install -U browser-use
python -m pip install playwright
playwright install
```

## Environment Settings

Set your OpenAI API key before running the examples:

```bash
export OPENAI_API_KEY="your API key"   # macOS/Linux
```

```powershell
$env:OPENAI_API_KEY="your API key"    # Windows (PowerShell)
```

## Usage

Create an agent and run a simple query. The example below opens Google, searches for "TEST",
and returns the first result.

```python
from browser_use import BrowserUse

browser = BrowserUse()
browser.run("Google検索でTESTと検索して最初の結果を取得")
```

### 自動スクリーンショット保存

環境変数でエージェント実行後にページスクリーンショットを自動保存できます。

```
export AUTO_SAVE_SCREENSHOTS=true           # 自動保存を有効化
export AUTO_SAVE_SCREENSHOTS_DIR=artifacts  # (任意) 保存先ディレクトリを変更
```

``AUTO_SAVE_SCREENSHOTS`` が有効な場合、`screenshots/`（または指定したディレクトリ）直下に
`agent_screenshot_YYYYMMDD_HHMMSS.png` 形式で保存されます。
