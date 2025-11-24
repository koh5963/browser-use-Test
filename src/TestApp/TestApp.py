"""Main module."""
import os
import asyncio
from browser_use.llm import ChatOpenAI
from browser_use.agent.service import Agent

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

    result = await agent.run()
    print("\n===== AGENT RESULT =====")
    print(result)
    
    llm.print_totals()

if __name__ == "__main__":
    asyncio.run(main())