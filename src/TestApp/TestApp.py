"""Main module."""
import os
import asyncio
import inspect
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from browser_use.agent.service import Agent
from browser_use.llm import ChatOpenAI

class LoggingChatOpenAI(ChatOpenAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 合計カウンタ
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0

    # ==== 同期呼び出し ====
    def invoke(self, *args, **kwargs):
        print("\n>>> [LLM] invoke called")
        resp = super().invoke(*args, **kwargs)
        self._handle_usage(resp)
        return resp

    def generate(self, *args, **kwargs):
        print("\n>>> [LLM] generate called")
        resp = super().generate(*args, **kwargs)
        self._handle_usage(resp)
        return resp

    # ==== 非同期呼び出し ====
    async def ainvoke(self, *args, **kwargs):
        print("\n>>> [LLM] ainvoke called")
        resp = await super().ainvoke(*args, **kwargs)
        self._handle_usage(resp)
        return resp

    async def agenerate(self, *args, **kwargs):
        print("\n>>> [LLM] agenerate called")
        resp = await super().agenerate(*args, **kwargs)
        self._handle_usage(resp)
        return resp

    # ==== usage 抽出 + カウンタ更新 ====
    def _handle_usage(self, resp):
        usage = self._extract_usage(resp)

        print(">>> [LLM] usage:", usage)

        if usage:
            inp = usage.get("input_tokens", 0) or 0
            out = usage.get("output_tokens", 0) or 0
            tot = usage.get("total_tokens", inp + out) or (inp + out)

            # 合計更新
            self.total_input_tokens += inp
            self.total_output_tokens += out
            self.total_tokens += tot

    # ==== usage抽出（あなたが書いた版を整理）====
    def _extract_usage(self, resp):
        usage = None

        # LangChain標準
        if hasattr(resp, "usage_metadata"):
            usage = resp.usage_metadata

        # OpenAI SDK (ChatCompletion 形式)
        if usage is None and hasattr(resp, "usage"):
            u = getattr(resp, "usage", None)
            if u:
                usage = {
                    "input_tokens": getattr(u, "prompt_tokens", None),
                    "output_tokens": getattr(u, "completion_tokens", None),
                    "total_tokens": getattr(u, "total_tokens", None),
                }

        # response_metadataパターン
        if usage is None and hasattr(resp, "response_metadata"):
            md = resp.response_metadata or {}
            usage = md.get("token_usage") or md.get("usage")

        # dict形式
        if usage is None and isinstance(resp, dict):
            usage = resp.get("usage") or resp.get("token_usage")

        return usage

    # ==== 合計を出力 ====
    def print_totals(self):
        print("\n===== TOTAL TOKEN USAGE =====")
        print("input_tokens :", self.total_input_tokens)
        print("output_tokens:", self.total_output_tokens)
        print("total_tokens :", self.total_tokens)
        print("================================\n")


async def main():
    await run_agent()

# FILE_URL = "http://localhost:8000/index.html"
FILE_URL = "http://localhost:8000/index2.html"

# Aパネル用タスク
# task = f"""
# 以下のローカルページを開いてください:
# {FILE_URL}

# ページ左上の A パネル「非決定的DOM差し替え」を対象にします。

# 手順:
# 1) 「商品コード」入力欄に ABC-123 を入力
# 2) 「チェック」ボタンを押す
# 3) 画面上に表示された結果メッセージ(✅/❌どちらでも)のテキストを教える

# 注意:
# - 結果DOMの出現位置が毎回変わるので、DOMセレクタに固執せず、表示文字を視覚的に探して抽出してください。
# """

task = f"""
あなたはブラウザ操作エージェントです。
次のローカルHTMLを開いてUI操作してください：
{FILE_URL}

画面には canvas 上に「氏名」「メール」「送信」ボタンが描かれています。
DOM上にinputやbuttonは存在しません。見た目（Vision）で判断してください。

手順:
1. 「氏名」右側の四角の入力エリアをクリックして, 表示されたalertダイアログに"Taro Yamada" を入力。
2. 「メール」右側の四角の入力エリアをクリックして, 表示されたalertダイアログに"taro@example.com" を入力。
3. 「送信」ボタンをクリック。
4. アラートに表示される送信内容を読み取り、氏名とメールを報告して終了。

注意:
- セレクタやDOM探索に頼らないこと
- クリック位置は見た目ベースで特定すること
- クリック位置の微調整が必要になる場合があること
- 入力欄は画面中央よりやや上側に縦に並んで二つ配置されていること
"""


async def run_agent():
    llm = LoggingChatOpenAI(   # ← ここを元のChatOpenAIから置き換える
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-5",
    )

    agent = Agent(
        llm=llm,
        task=task,
        max_actions_per_step=3,
        max_steps=10,
        max_failures=2,
        directly_open_url=True,
        step_timeout=120,
        llm_timeout=120,
        use_vision=True,
        vision_detail_level="low",
    )

    patch_vision_click_with_js_offset(agent)
    install_dialog_handler(agent)

    result = await agent.run()
    print("\n===== AGENT RESULT =====")
    print(result)

    if should_auto_save_screenshots():
        await save_screenshot(agent)

    llm.print_totals()


def patch_vision_click_with_js_offset(agent: Agent) -> None:
    """Visionクリック時の座標にdevicePixelRatio補正をかけるJSラッパーを登録する。

    browser-useのVisionはスクリーンショット座標を返すため、devicePixelRatioを考慮しないと
    実ページのクリック位置がずれることがある。click_on_coordinates相当のメソッドをラップし、
    JSでCSSピクセルに補正したうえで既存のクリック処理を呼び出す。
    """

    browser = getattr(agent, "browser", None)
    controller = getattr(browser, "playwright_controller", None) or getattr(browser, "_playwright_controller", None)
    if controller is None:
        return

    click_fn = _resolve_click_function(controller)
    if click_fn is None or getattr(controller, "_vision_click_wrapped", False):
        return

    signature = inspect.signature(click_fn)

    async def _wrapped_click(*args: Any, **kwargs: Any):
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        x = bound.arguments.get("x") or bound.arguments.get("coord_x")
        y = bound.arguments.get("y") or bound.arguments.get("coord_y")
        page = bound.arguments.get("page") or getattr(controller, "page", None)

        if page is not None and x is not None and y is not None:
            corrected = await page.evaluate(
                """
                ({ x, y }) => {
                    const ratio = window.devicePixelRatio || 1;

                    // 視覚ビューポートのズーム補正（ピンチズーム等）。
                    const vv = window.visualViewport;
                    const viewportScale = vv?.scale ?? 1;
                    const viewportOffsetX = (vv?.offsetLeft ?? 0) + (vv?.pageLeft ?? 0);
                    const viewportOffsetY = (vv?.offsetTop ?? 0) + (vv?.pageTop ?? 0);

                    const scrollX = window.scrollX || 0;
                    const scrollY = window.scrollY || 0;

                    // devicePixelRatio と viewport のズーム倍率を両方考慮する。
                    const scale = ratio * viewportScale;

                    return {
                        x: x / scale + scrollX + viewportOffsetX,
                        y: y / scale + scrollY + viewportOffsetY,
                    };
                }
                """,
                {"x": x, "y": y},
            )

            bound.arguments["x"] = corrected.get("x", x)
            bound.arguments["y"] = corrected.get("y", y)

        return await click_fn(*bound.args, **bound.kwargs)

    setattr(controller, click_fn.__name__, _wrapped_click)
    setattr(controller, "_vision_click_wrapped", True)


def _resolve_click_function(controller: Any) -> Optional[Callable[..., Any]]:
    for candidate in ("click_on_coordinates", "click_on_page_coordinates", "click"):
        if hasattr(controller, candidate):
            fn = getattr(controller, candidate)
            if callable(fn):
                return fn
    return None


def install_dialog_handler(agent: Agent) -> None:
    """Playwrightダイアログ(alert/prompt)を可視化・自動応答する。

    headless環境ではポップアップが視覚的に出ないため、dialogイベントをフックして
    - 受信したダイアログをページ右上にログ表示
    - promptには内容に応じた自動入力を行う
    を行い、エージェントのクリック後にアラートが見えない問題を防ぐ。
    """

    browser = getattr(agent, "browser", None)
    controller = getattr(browser, "playwright_controller", None) or getattr(browser, "_playwright_controller", None)
    if controller is None or getattr(controller, "_dialog_handler_installed", False):
        return

    page = getattr(controller, "page", None)
    if page is None:
        return

    async def _on_dialog(dialog):
        message = dialog.message
        dialog_type = dialog.type

        # promptの入力値を決定（氏名/メールに応じて自動入力）
        response = None
        if dialog_type == "prompt":
            if "氏名" in message:
                response = "Taro Yamada"
            elif "メール" in message:
                response = "taro@example.com"
            else:
                response = dialog.defaultValue or ""
            await dialog.accept(response)
        else:
            await dialog.accept()

        # 画面右上にログ表示してVisionで読めるようにする
        await page.evaluate(
            """
            (msg, type) => {
                const id = "dialog-log-overlay";
                let box = document.getElementById(id);
                if (!box) {
                    box = document.createElement("div");
                    box.id = id;
                    Object.assign(box.style, {
                        position: "fixed",
                        top: "16px",
                        right: "16px",
                        padding: "12px 14px",
                        background: "rgba(0, 0, 0, 0.78)",
                        color: "#fff",
                        fontSize: "14px",
                        zIndex: 99999,
                        maxWidth: "420px",
                        borderRadius: "10px",
                        boxShadow: "0 8px 18px rgba(0,0,0,0.35)",
                        lineHeight: "1.4",
                        backdropFilter: "blur(2px)",
                        whiteSpace: "pre-wrap",
                    });
                    document.body.appendChild(box);
                }

                // エージェントが後段で読めるように最新ダイアログを window に保持
                const store = (window.__dialogMessages ||= []);
                const payload = { type, message: msg, timestamp: Date.now() };
                store.push(payload);
                window.__lastDialogMessage = payload;
                const line = document.createElement("div");
                line.textContent = `[${type}] ${msg}`;
                line.style.marginBottom = "6px";
                box.appendChild(line);
            }
            """,
            message,
            dialog_type,
        )

    page.on("dialog", _on_dialog)
    setattr(controller, "_dialog_handler_installed", True)


def should_auto_save_screenshots() -> bool:
    """環境変数に基づきスクリーンショット自動保存を有効化するか判定する。"""

    flag = os.environ.get("AUTO_SAVE_SCREENSHOTS", "false").lower()
    return flag in {"1", "true", "yes", "on"}


async def save_screenshot(agent: Agent, directory: Optional[str] = None) -> Optional[Path]:
    """Playwrightページのスクリーンショットを保存する。

    Args:
        agent: 実行中のAgentインスタンス。
        directory: 保存先ディレクトリ。未指定の場合は ``AUTO_SAVE_SCREENSHOTS_DIR``
            環境変数、もしくは ``screenshots`` を利用する。

    Returns:
        保存されたファイルパス（失敗時は None）。
    """

    target_dir = Path(directory or os.environ.get("AUTO_SAVE_SCREENSHOTS_DIR", "screenshots"))
    target_dir.mkdir(parents=True, exist_ok=True)

    browser = getattr(agent, "browser", None)
    controller = getattr(browser, "playwright_controller", None) or getattr(browser, "_playwright_controller", None)
    page = getattr(controller, "page", None) if controller else None

    if page is None:
        print(">>> [Screenshot] ページオブジェクトが見つからないためスクリーンショットを保存できませんでした")
        return None

    filename = f"agent_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = target_dir / filename

    try:
        await page.screenshot(path=str(filepath), full_page=True)
        print(f">>> [Screenshot] スクリーンショットを保存しました: {filepath}")
        return filepath
    except Exception as exc:  # noqa: BLE001 - トレース用に例外をそのまま表示
        print(f">>> [Screenshot] スクリーンショットの保存に失敗しました: {exc}")
        return None

if __name__ == "__main__":
    asyncio.run(main())
