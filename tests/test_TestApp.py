#!/usr/bin/env python
"""Tests for `TestApp` package."""
import os
import unittest.mock
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, call, patch

import pytest

from TestApp.TestApp import (
    LoggingChatOpenAI,
    _resolve_click_function,
    _resolve_playwright_page,
    install_dialog_handler,
    patch_vision_click_with_js_offset,
    save_screenshot,
    should_auto_save_screenshots,
)


# ============================================================================
# LoggingChatOpenAI Tests
# ============================================================================


class TestLoggingChatOpenAI:
    """Test suite for LoggingChatOpenAI class."""

    def test_init_initializes_token_counters(self):
        """Test that __init__ properly initializes token counters to zero."""
        with patch("TestApp.TestApp.ChatOpenAI.__init__", return_value=None):
            llm = LoggingChatOpenAI(api_key="test-key")
            assert llm.total_input_tokens == 0
            assert llm.total_output_tokens == 0
            assert llm.total_tokens == 0

    def test_invoke_calls_parent_and_handles_usage(self, capsys):
        """Test that invoke calls parent method and processes usage data."""
        with patch("TestApp.TestApp.ChatOpenAI.__init__", return_value=None):
            llm = LoggingChatOpenAI()
            mock_response = Mock()
            mock_response.usage_metadata = {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }

            with patch("TestApp.TestApp.ChatOpenAI.invoke", return_value=mock_response):
                result = llm.invoke("test prompt")

            assert result == mock_response
            assert llm.total_input_tokens == 100
            assert llm.total_output_tokens == 50
            assert llm.total_tokens == 150

            captured = capsys.readouterr()
            assert "[LLM] invoke called" in captured.out
            assert "[LLM] usage:" in captured.out

    def test_generate_calls_parent_and_handles_usage(self, capsys):
        """Test that generate calls parent method and processes usage data."""
        with patch("TestApp.TestApp.ChatOpenAI.__init__", return_value=None):
            llm = LoggingChatOpenAI()
            mock_response = Mock()
            mock_response.usage_metadata = {
                "input_tokens": 200,
                "output_tokens": 75,
                "total_tokens": 275,
            }

            with patch("TestApp.TestApp.ChatOpenAI.generate", return_value=mock_response):
                result = llm.generate("test prompt")

            assert result == mock_response
            assert llm.total_input_tokens == 200
            assert llm.total_output_tokens == 75
            assert llm.total_tokens == 275

            captured = capsys.readouterr()
            assert "[LLM] generate called" in captured.out

    @pytest.mark.asyncio
    async def test_ainvoke_calls_parent_and_handles_usage(self, capsys):
        """Test that ainvoke calls parent async method and processes usage data."""
        with patch("TestApp.TestApp.ChatOpenAI.__init__", return_value=None):
            llm = LoggingChatOpenAI()
            mock_response = Mock()
            mock_response.usage_metadata = {
                "input_tokens": 150,
                "output_tokens": 100,
                "total_tokens": 250,
            }

            with patch("TestApp.TestApp.ChatOpenAI.ainvoke", new_callable=AsyncMock, return_value=mock_response):
                result = await llm.ainvoke("test prompt")

            assert result == mock_response
            assert llm.total_input_tokens == 150
            assert llm.total_output_tokens == 100
            assert llm.total_tokens == 250

            captured = capsys.readouterr()
            assert "[LLM] ainvoke called" in captured.out

    @pytest.mark.asyncio
    async def test_agenerate_calls_parent_and_handles_usage(self, capsys):
        """Test that agenerate calls parent async method and processes usage data."""
        with patch("TestApp.TestApp.ChatOpenAI.__init__", return_value=None):
            llm = LoggingChatOpenAI()
            mock_response = Mock()
            mock_response.usage_metadata = {
                "input_tokens": 300,
                "output_tokens": 200,
                "total_tokens": 500,
            }

            with patch("TestApp.TestApp.ChatOpenAI.agenerate", new_callable=AsyncMock, return_value=mock_response):
                result = await llm.agenerate("test prompt")

            assert result == mock_response
            assert llm.total_input_tokens == 300
            assert llm.total_output_tokens == 200
            assert llm.total_tokens == 500

            captured = capsys.readouterr()
            assert "[LLM] agenerate called" in captured.out

    def test_handle_usage_accumulates_tokens_across_calls(self):
        """Test that multiple calls accumulate token counts correctly."""
        with patch("TestApp.TestApp.ChatOpenAI.__init__", return_value=None):
            llm = LoggingChatOpenAI()

            # First call
            response1 = Mock()
            response1.usage_metadata = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
            llm._handle_usage(response1)

            # Second call
            response2 = Mock()
            response2.usage_metadata = {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300}
            llm._handle_usage(response2)

            assert llm.total_input_tokens == 300
            assert llm.total_output_tokens == 150
            assert llm.total_tokens == 450

    def test_extract_usage_with_usage_metadata(self):
        """Test _extract_usage with usage_metadata attribute."""
        with patch("TestApp.TestApp.ChatOpenAI.__init__", return_value=None):
            llm = LoggingChatOpenAI()
            response = Mock()
            response.usage_metadata = {"input_tokens": 100, "output_tokens": 50}

            usage = llm._extract_usage(response)
            assert usage == {"input_tokens": 100, "output_tokens": 50}

    def test_extract_usage_with_openai_sdk_format(self):
        """Test _extract_usage with OpenAI SDK usage format."""
        with patch("TestApp.TestApp.ChatOpenAI.__init__", return_value=None):
            llm = LoggingChatOpenAI()
            response = Mock()
            delattr(response, "usage_metadata")
            response.usage = Mock(prompt_tokens=100, completion_tokens=50, total_tokens=150)

            usage = llm._extract_usage(response)
            assert usage == {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}

    def test_extract_usage_with_response_metadata(self):
        """Test _extract_usage with response_metadata.token_usage format."""
        with patch("TestApp.TestApp.ChatOpenAI.__init__", return_value=None):
            llm = LoggingChatOpenAI()
            response = Mock()
            delattr(response, "usage_metadata")
            delattr(response, "usage")
            response.response_metadata = {"token_usage": {"input_tokens": 75, "output_tokens": 25}}

            usage = llm._extract_usage(response)
            assert usage == {"input_tokens": 75, "output_tokens": 25}

    def test_extract_usage_with_dict_format(self):
        """Test _extract_usage with dict response format."""
        with patch("TestApp.TestApp.ChatOpenAI.__init__", return_value=None):
            llm = LoggingChatOpenAI()
            response = {"usage": {"input_tokens": 60, "output_tokens": 40}}

            usage = llm._extract_usage(response)
            assert usage == {"input_tokens": 60, "output_tokens": 40}

    def test_extract_usage_handles_none_values(self):
        """Test that _extract_usage handles None values gracefully."""
        with patch("TestApp.TestApp.ChatOpenAI.__init__", return_value=None):
            llm = LoggingChatOpenAI()
            response = Mock()
            response.usage_metadata = {"input_tokens": None, "output_tokens": 50, "total_tokens": None}

            llm._handle_usage(response)
            assert llm.total_input_tokens == 0
            assert llm.total_output_tokens == 50
            assert llm.total_tokens == 50

    def test_extract_usage_calculates_total_from_input_output(self):
        """Test that total_tokens is calculated when not provided."""
        with patch("TestApp.TestApp.ChatOpenAI.__init__", return_value=None):
            llm = LoggingChatOpenAI()
            response = Mock()
            response.usage_metadata = {"input_tokens": 100, "output_tokens": 50}

            llm._handle_usage(response)
            assert llm.total_tokens == 150

    def test_extract_usage_returns_none_when_no_usage_found(self):
        """Test that _extract_usage returns None when no usage data found."""
        with patch("TestApp.TestApp.ChatOpenAI.__init__", return_value=None):
            llm = LoggingChatOpenAI()
            response = Mock(spec=[])

            usage = llm._extract_usage(response)
            assert usage is None

    def test_print_totals_displays_correct_format(self, capsys):
        """Test that print_totals displays token counts in correct format."""
        with patch("TestApp.TestApp.ChatOpenAI.__init__", return_value=None):
            llm = LoggingChatOpenAI()
            llm.total_input_tokens = 1000
            llm.total_output_tokens = 500
            llm.total_tokens = 1500

            llm.print_totals()

            captured = capsys.readouterr()
            assert "===== TOTAL TOKEN USAGE =====" in captured.out
            assert "input_tokens : 1000" in captured.out
            assert "output_tokens: 500" in captured.out
            assert "total_tokens : 1500" in captured.out


# ============================================================================
# patch_vision_click_with_js_offset Tests
# ============================================================================


class TestPatchVisionClickWithJSOffset:
    """Test suite for patch_vision_click_with_js_offset function."""

    def test_returns_early_when_controller_is_none(self):
        """Test that function returns early if controller cannot be found."""
        mock_agent = Mock()
        mock_agent.browser = Mock()
        mock_agent.browser.playwright_controller = None
        mock_agent.browser._playwright_controller = None
        patch_vision_click_with_js_offset(mock_agent)

    def test_returns_early_when_browser_is_none(self):
        """Test that function returns early if browser is None."""
        mock_agent = Mock()
        mock_agent.browser = None
        patch_vision_click_with_js_offset(mock_agent)

    def test_returns_early_when_click_function_not_found(self):
        """Test that function returns early if no click function found."""
        mock_agent = Mock()
        mock_controller = Mock(spec=[])
        mock_agent.browser = Mock()
        mock_agent.browser.playwright_controller = mock_controller
        patch_vision_click_with_js_offset(mock_agent)

    def test_returns_early_when_already_wrapped(self):
        """Test that function returns early if already wrapped."""
        mock_agent = Mock()
        mock_controller = Mock()
        mock_controller._vision_click_wrapped = True
        mock_controller.click_on_coordinates = Mock()
        mock_agent.browser = Mock()
        mock_agent.browser.playwright_controller = mock_controller

        original_click = mock_controller.click_on_coordinates
        patch_vision_click_with_js_offset(mock_agent)
        assert mock_controller.click_on_coordinates == original_click

    @pytest.mark.asyncio
    async def test_wraps_click_on_coordinates_function(self):
        """Test that click_on_coordinates is properly wrapped."""
        mock_agent = Mock()
        mock_controller = Mock()
        mock_controller._vision_click_wrapped = False
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value={"x": 100, "y": 200})
        mock_controller.page = mock_page

        async def mock_click(x, y, page=None):
            return {"clicked": True, "x": x, "y": y}

        mock_controller.click_on_coordinates = mock_click
        mock_agent.browser = Mock()
        mock_agent.browser.playwright_controller = mock_controller

        patch_vision_click_with_js_offset(mock_agent)
        assert mock_controller._vision_click_wrapped is True
        assert hasattr(mock_controller, "click_on_coordinates")

        result = await mock_controller.click_on_coordinates(x=50, y=100, page=mock_page)
        assert result is not None

    @pytest.mark.asyncio
    async def test_wrapped_click_corrects_coordinates_with_page_evaluate(self):
        """Test that wrapped click function calls page.evaluate to correct coordinates."""
        mock_agent = Mock()
        mock_controller = Mock()
        mock_controller._vision_click_wrapped = False
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value={"x": 100.5, "y": 200.5})

        async def mock_click(x, y, page=None):
            return {"x": x, "y": y}

        mock_controller.click_on_coordinates = mock_click
        mock_controller.page = mock_page
        mock_agent.browser = Mock()
        mock_agent.browser.playwright_controller = mock_controller

        patch_vision_click_with_js_offset(mock_agent)
        result = await mock_controller.click_on_coordinates(x=50, y=100, page=mock_page)

        mock_page.evaluate.assert_called_once()
        call_args = mock_page.evaluate.call_args
        assert "devicePixelRatio" in call_args[0][0]
        assert call_args[0][1] == {"x": 50, "y": 100}
        assert result["x"] == 100.5
        assert result["y"] == 200.5


# ============================================================================
# _resolve_click_function Tests
# ============================================================================


class TestResolveClickFunction:
    """Test suite for _resolve_click_function helper function."""

    def test_returns_click_on_coordinates(self):
        """Test that it finds and returns click_on_coordinates."""
        controller = Mock()
        controller.click_on_coordinates = Mock()
        result = _resolve_click_function(controller)
        assert result == controller.click_on_coordinates

    def test_returns_click_on_page_coordinates(self):
        """Test that it finds and returns click_on_page_coordinates."""
        controller = Mock(spec=["click_on_page_coordinates"])
        controller.click_on_page_coordinates = Mock()
        result = _resolve_click_function(controller)
        assert result == controller.click_on_page_coordinates

    def test_returns_click_as_fallback(self):
        """Test that it returns click method as fallback."""
        controller = Mock(spec=["click"])
        controller.click = Mock()
        result = _resolve_click_function(controller)
        assert result == controller.click

    def test_returns_none_when_no_click_function_found(self):
        """Test that it returns None when no click function exists."""
        controller = Mock(spec=[])
        result = _resolve_click_function(controller)
        assert result is None

    def test_returns_none_when_attribute_not_callable(self):
        """Test that it returns None when attribute exists but is not callable."""
        controller = Mock()
        controller.click_on_coordinates = "not_a_function"
        result = _resolve_click_function(controller)
        assert result is None


# ============================================================================
# install_dialog_handler Tests
# ============================================================================


class TestInstallDialogHandler:
    """Test suite for install_dialog_handler function."""

    def test_returns_early_when_controller_is_none(self):
        """Test that function returns early if controller is None."""
        mock_agent = Mock()
        mock_agent.browser = Mock()
        mock_agent.browser.playwright_controller = None
        mock_agent.browser._playwright_controller = None
        install_dialog_handler(mock_agent)

    def test_returns_early_when_already_installed(self):
        """Test that function returns early if handler already installed."""
        mock_agent = Mock()
        mock_controller = Mock()
        mock_controller._dialog_handler_installed = True
        mock_controller.page = Mock()
        mock_agent.browser = Mock()
        mock_agent.browser.playwright_controller = mock_controller

        install_dialog_handler(mock_agent)
        mock_controller.page.on.assert_not_called()

    def test_returns_early_when_page_is_none(self):
        """Test that function returns early if page is None."""
        mock_agent = Mock()
        mock_controller = Mock()
        mock_controller._dialog_handler_installed = False
        mock_controller.page = None
        mock_agent.browser = Mock()
        mock_agent.browser.playwright_controller = mock_controller
        install_dialog_handler(mock_agent)

    def test_installs_dialog_handler_on_page(self):
        """Test that dialog handler is installed on page."""
        mock_agent = Mock()
        mock_controller = Mock()
        mock_controller._dialog_handler_installed = False
        mock_page = Mock()
        mock_controller.page = mock_page
        mock_agent.browser = Mock()
        mock_agent.browser.playwright_controller = mock_controller

        install_dialog_handler(mock_agent)
        mock_page.on.assert_called_once_with("dialog", unittest.mock.ANY)
        assert mock_controller._dialog_handler_installed is True

    @pytest.mark.asyncio
    async def test_dialog_handler_accepts_alert(self):
        """Test that dialog handler accepts alert dialogs."""
        mock_agent = Mock()
        mock_controller = Mock()
        mock_controller._dialog_handler_installed = False
        mock_page = AsyncMock()
        mock_controller.page = mock_page
        mock_agent.browser = Mock()
        mock_agent.browser.playwright_controller = mock_controller

        install_dialog_handler(mock_agent)
        handler = mock_page.on.call_args[0][1]

        mock_dialog = AsyncMock()
        mock_dialog.message = "This is an alert"
        mock_dialog.type = "alert"

        await handler(mock_dialog)
        mock_dialog.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_dialog_handler_accepts_prompt_with_name(self):
        """Test that dialog handler provides name for 氏名 prompt."""
        mock_agent = Mock()
        mock_controller = Mock()
        mock_controller._dialog_handler_installed = False
        mock_page = AsyncMock()
        mock_controller.page = mock_page
        mock_agent.browser = Mock()
        mock_agent.browser.playwright_controller = mock_controller

        install_dialog_handler(mock_agent)
        handler = mock_page.on.call_args[0][1]

        mock_dialog = AsyncMock()
        mock_dialog.message = "氏名を入力してください"
        mock_dialog.type = "prompt"

        await handler(mock_dialog)
        mock_dialog.accept.assert_called_once_with("Taro Yamada")

    @pytest.mark.asyncio
    async def test_dialog_handler_accepts_prompt_with_email(self):
        """Test that dialog handler provides email for メール prompt."""
        mock_agent = Mock()
        mock_controller = Mock()
        mock_controller._dialog_handler_installed = False
        mock_page = AsyncMock()
        mock_controller.page = mock_page
        mock_agent.browser = Mock()
        mock_agent.browser.playwright_controller = mock_controller

        install_dialog_handler(mock_agent)
        handler = mock_page.on.call_args[0][1]

        mock_dialog = AsyncMock()
        mock_dialog.message = "メールアドレスを入力"
        mock_dialog.type = "prompt"

        await handler(mock_dialog)
        mock_dialog.accept.assert_called_once_with("taro@example.com")

    @pytest.mark.asyncio
    async def test_dialog_handler_uses_default_value_for_unknown_prompt(self):
        """Test that dialog handler uses default value for unknown prompt."""
        mock_agent = Mock()
        mock_controller = Mock()
        mock_controller._dialog_handler_installed = False
        mock_page = AsyncMock()
        mock_controller.page = mock_page
        mock_agent.browser = Mock()
        mock_agent.browser.playwright_controller = mock_controller

        install_dialog_handler(mock_agent)
        handler = mock_page.on.call_args[0][1]

        mock_dialog = AsyncMock()
        mock_dialog.message = "Enter some value"
        mock_dialog.type = "prompt"
        mock_dialog.defaultValue = "default text"

        await handler(mock_dialog)
        mock_dialog.accept.assert_called_once_with("default text")

    @pytest.mark.asyncio
    async def test_dialog_handler_displays_dialog_log_overlay(self):
        """Test that dialog handler displays log overlay on page."""
        mock_agent = Mock()
        mock_controller = Mock()
        mock_controller._dialog_handler_installed = False
        mock_page = AsyncMock()
        mock_controller.page = mock_page
        mock_agent.browser = Mock()
        mock_agent.browser.playwright_controller = mock_controller

        install_dialog_handler(mock_agent)
        handler = mock_page.on.call_args[0][1]

        mock_dialog = AsyncMock()
        mock_dialog.message = "Test dialog"
        mock_dialog.type = "alert"

        await handler(mock_dialog)
        mock_page.evaluate.assert_called_once()
        call_args = mock_page.evaluate.call_args
        assert "dialog-log-overlay" in call_args[0][0]
        assert call_args[0][1] == "Test dialog"
        assert call_args[0][2] == "alert"


# ============================================================================
# should_auto_save_screenshots Tests
# ============================================================================


class TestShouldAutoSaveScreenshots:
    """Test suite for should_auto_save_screenshots function."""

    def test_returns_false_when_env_not_set(self):
        """Test that it returns False when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            assert should_auto_save_screenshots() is False

    def test_returns_false_for_false_string(self):
        """Test that it returns False for 'false' string."""
        with patch.dict(os.environ, {"AUTO_SAVE_SCREENSHOTS": "false"}):
            assert should_auto_save_screenshots() is False

    def test_returns_true_for_true_string(self):
        """Test that it returns True for 'true' string."""
        with patch.dict(os.environ, {"AUTO_SAVE_SCREENSHOTS": "true"}):
            assert should_auto_save_screenshots() is True

    def test_returns_true_for_1_string(self):
        """Test that it returns True for '1' string."""
        with patch.dict(os.environ, {"AUTO_SAVE_SCREENSHOTS": "1"}):
            assert should_auto_save_screenshots() is True

    def test_returns_true_for_yes_string(self):
        """Test that it returns True for 'yes' string."""
        with patch.dict(os.environ, {"AUTO_SAVE_SCREENSHOTS": "yes"}):
            assert should_auto_save_screenshots() is True

    def test_returns_true_for_on_string(self):
        """Test that it returns True for 'on' string."""
        with patch.dict(os.environ, {"AUTO_SAVE_SCREENSHOTS": "on"}):
            assert should_auto_save_screenshots() is True

    def test_is_case_insensitive(self):
        """Test that the check is case-insensitive."""
        with patch.dict(os.environ, {"AUTO_SAVE_SCREENSHOTS": "TRUE"}):
            assert should_auto_save_screenshots() is True
        with patch.dict(os.environ, {"AUTO_SAVE_SCREENSHOTS": "Yes"}):
            assert should_auto_save_screenshots() is True

    def test_returns_false_for_invalid_string(self):
        """Test that it returns False for invalid string values."""
        with patch.dict(os.environ, {"AUTO_SAVE_SCREENSHOTS": "maybe"}):
            assert should_auto_save_screenshots() is False


# ============================================================================
# save_screenshot Tests
# ============================================================================


class TestSaveScreenshot:
    """Test suite for save_screenshot function."""

    @pytest.mark.asyncio
    async def test_creates_directory_if_not_exists(self, tmp_path):
        """Test that the function creates the screenshot directory if it doesn't exist."""
        screenshot_dir = tmp_path / "test_screenshots"
        mock_agent = Mock()
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock()

        mock_controller = Mock()
        mock_controller.page = mock_page
        mock_agent.browser = Mock()
        mock_agent.browser.playwright_controller = mock_controller

        await save_screenshot(mock_agent, directory=str(screenshot_dir))
        assert screenshot_dir.exists()

    @pytest.mark.asyncio
    async def test_uses_default_directory_when_not_specified(self, tmp_path):
        """Test that function uses default directory from env or 'screenshots'."""
        with patch.dict(os.environ, {"AUTO_SAVE_SCREENSHOTS_DIR": str(tmp_path / "custom")}):
            mock_agent = Mock()
            mock_page = AsyncMock()
            mock_page.screenshot = AsyncMock()

            mock_controller = Mock()
            mock_controller.page = mock_page
            mock_agent.browser = Mock()
            mock_agent.browser.playwright_controller = mock_controller

            await save_screenshot(mock_agent)
            expected_dir = tmp_path / "custom"
            assert expected_dir.exists()

    @pytest.mark.asyncio
    async def test_saves_screenshot_with_timestamp_filename(self, tmp_path):
        """Test that screenshot is saved with timestamped filename."""
        screenshot_dir = tmp_path / "screenshots"
        mock_agent = Mock()
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock()

        mock_controller = Mock()
        mock_controller.page = mock_page
        mock_agent.browser = Mock()
        mock_agent.browser.playwright_controller = mock_controller

        with patch("TestApp.TestApp.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20231225_123456"
            result = await save_screenshot(mock_agent, directory=str(screenshot_dir))

            expected_path = screenshot_dir / "agent_screenshot_20231225_123456.png"
            assert result == expected_path
            mock_page.screenshot.assert_called_once_with(path=str(expected_path), full_page=True)

    @pytest.mark.asyncio
    async def test_returns_none_when_page_not_found(self, tmp_path, capsys):
        """Test that function returns None when page cannot be resolved."""
        screenshot_dir = tmp_path / "screenshots"
        mock_agent = Mock()

        mock_controller = Mock()
        mock_controller.page = None
        mock_agent.browser = Mock()
        mock_agent.browser.playwright_controller = mock_controller

        result = await save_screenshot(mock_agent, directory=str(screenshot_dir))
        assert result is None
        captured = capsys.readouterr()
        assert "ページオブジェクトが見つからない" in captured.out

    @pytest.mark.asyncio
    async def test_returns_none_on_screenshot_exception(self, tmp_path, capsys):
        """Test that function returns None and prints error on exception."""
        screenshot_dir = tmp_path / "screenshots"
        mock_agent = Mock()
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock(side_effect=Exception("Screenshot failed"))

        mock_controller = Mock()
        mock_controller.page = mock_page
        mock_agent.browser = Mock()
        mock_agent.browser.playwright_controller = mock_controller

        result = await save_screenshot(mock_agent, directory=str(screenshot_dir))
        assert result is None
        captured = capsys.readouterr()
        assert "スクリーンショットの保存に失敗" in captured.out


# ============================================================================
# _resolve_playwright_page Tests
# ============================================================================


class TestResolvePlaywrightPage:
    """Test suite for _resolve_playwright_page function."""

    def test_returns_none_when_controller_is_none(self):
        """Test that function returns None when controller is None."""
        result = _resolve_playwright_page(None)
        assert result is None

    def test_returns_page_from_controller_page_attribute(self):
        """Test that function returns page from controller.page if available."""
        mock_controller = Mock()
        mock_page = Mock()
        mock_controller.page = mock_page
        result = _resolve_playwright_page(mock_controller)
        assert result == mock_page

    def test_returns_page_from_context_pages(self):
        """Test that function returns page from context.pages when controller.page is None."""
        mock_controller = Mock()
        mock_controller.page = None
        mock_page = Mock()
        mock_controller.context = Mock()
        mock_controller.context.pages = [mock_page]
        result = _resolve_playwright_page(mock_controller)
        assert result == mock_page

    def test_returns_page_from_context_pages_callable(self):
        """Test that function handles callable context.pages()."""
        mock_controller = Mock()
        mock_controller.page = None
        mock_page = Mock()
        mock_controller.context = Mock()
        mock_controller.context.pages = Mock(return_value=[mock_page])
        result = _resolve_playwright_page(mock_controller)
        assert result == mock_page

    def test_returns_page_from_browser_contexts(self):
        """Test that function returns page from browser.contexts when context is None."""
        mock_controller = Mock()
        mock_controller.page = None
        mock_controller.context = None
        mock_page = Mock()
        mock_context = Mock()
        mock_context.pages = Mock(return_value=[mock_page])
        mock_controller.browser = Mock()
        mock_controller.browser.contexts = Mock(return_value=[mock_context])
        result = _resolve_playwright_page(mock_controller)
        assert result == mock_page

    def test_returns_page_from_browser_contexts_attribute(self):
        """Test that function handles non-callable browser.contexts."""
        mock_controller = Mock()
        mock_controller.page = None
        mock_controller.context = None
        mock_page = Mock()
        mock_context = Mock()
        mock_context.pages = [mock_page]
        mock_controller.browser = Mock()
        mock_controller.browser.contexts = [mock_context]
        result = _resolve_playwright_page(mock_controller)
        assert result == mock_page

    def test_returns_none_when_no_page_found(self):
        """Test that function returns None when no page can be found."""
        mock_controller = Mock()
        mock_controller.page = None
        mock_controller.context = None
        mock_controller.browser = None
        result = _resolve_playwright_page(mock_controller)
        assert result is None

    def test_returns_none_when_context_has_no_pages(self):
        """Test that function returns None when context has empty pages list."""
        mock_controller = Mock()
        mock_controller.page = None
        mock_controller.context = Mock()
        mock_controller.context.pages = []
        result = _resolve_playwright_page(mock_controller)
        assert result is None

    def test_iterates_through_multiple_contexts(self):
        """Test that function iterates through multiple contexts to find page."""
        mock_controller = Mock()
        mock_controller.page = None
        mock_controller.context = None

        mock_page = Mock()
        mock_context1 = Mock()
        mock_context1.pages = Mock(return_value=[])
        mock_context2 = Mock()
        mock_context2.pages = Mock(return_value=[mock_page])

        mock_controller.browser = Mock()
        mock_controller.browser.contexts = Mock(return_value=[mock_context1, mock_context2])

        result = _resolve_playwright_page(mock_controller)
        assert result == mock_page


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for combined functionality."""

    @pytest.mark.asyncio
    async def test_logging_llm_with_multiple_sequential_calls(self):
        """Test LoggingChatOpenAI accumulates tokens across multiple calls."""
        with patch("TestApp.TestApp.ChatOpenAI.__init__", return_value=None):
            llm = LoggingChatOpenAI()

            responses = [
                Mock(usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}),
                Mock(usage_metadata={"input_tokens": 200, "output_tokens": 75, "total_tokens": 275}),
                Mock(usage_metadata={"input_tokens": 150, "output_tokens": 100, "total_tokens": 250}),
            ]

            with patch("TestApp.TestApp.ChatOpenAI.invoke", side_effect=responses):
                for _ in range(3):
                    llm.invoke("test")

            assert llm.total_input_tokens == 450
            assert llm.total_output_tokens == 225
            assert llm.total_tokens == 675

    @pytest.mark.asyncio
    async def test_screenshot_with_auto_save_enabled(self, tmp_path):
        """Test screenshot saving when auto-save is enabled."""
        screenshot_dir = tmp_path / "screenshots"

        with patch.dict(os.environ, {"AUTO_SAVE_SCREENSHOTS": "true"}):
            assert should_auto_save_screenshots() is True

            mock_agent = Mock()
            mock_page = AsyncMock()
            mock_page.screenshot = AsyncMock()

            mock_controller = Mock()
            mock_controller.page = mock_page
            mock_agent.browser = Mock()
            mock_agent.browser.playwright_controller = mock_controller

            result = await save_screenshot(mock_agent, directory=str(screenshot_dir))
            assert result is not None
            assert result.exists()
