# Browser-Use-Test

~~![PyPI version](https://img.shields.io/pypi/v/TestApp.svg)~~
~~[![Documentation Status](https://readthedocs.org/projects/TestApp/badge/?version=latest)](https://TestApp.readthedocs.io/en/latest/?version=latest)~~

Application for ***browser-use*** practice.  
This repository is for experimenting with browser automation + Vision-based UI analysis.  

* Free software: MIT License
* ~~PyPI package: https://pypi.org/project/TestApp/~~
* ~~Documentation: https://TestApp.readthedocs.io.~~

## Features

* browser-useもろもろ検証
* Vision機能検証
* 将来的には仕様書を読み込んでテストケース自動生成→自動テスト→エビデンス作成まで

## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.

## Installation

```powershell
pip install -U browser-use
pip install playwright
playwright install
```

## Environment Settings
```powershell
$env:OPENAI_API_KEY="your API key"
```

## Usage

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
