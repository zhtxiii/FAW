"""tests/test_tools.py - 技能系统测试。"""

import pytest
from skill_manager import SkillManager, create_default_registry
from skills.calculator import CalculatorSkill
from skills.text_processor import TextProcessorSkill


class TestCalculatorSkill:
    @pytest.fixture
    def calc(self):
        return CalculatorSkill()

    @pytest.mark.asyncio
    async def test_addition(self, calc):
        result = await calc.execute({"expression": "2 + 3"})
        assert result["result"] == 5

    @pytest.mark.asyncio
    async def test_complex_expression(self, calc):
        result = await calc.execute({"expression": "(10 + 5) * 3 - 2"})
        assert result["result"] == 43

    @pytest.mark.asyncio
    async def test_division(self, calc):
        result = await calc.execute({"expression": "10 / 4"})
        assert result["result"] == 2.5

    @pytest.mark.asyncio
    async def test_power(self, calc):
        result = await calc.execute({"expression": "2 ** 10"})
        assert result["result"] == 1024

    @pytest.mark.asyncio
    async def test_negative(self, calc):
        result = await calc.execute({"expression": "-5 + 3"})
        assert result["result"] == -2

    @pytest.mark.asyncio
    async def test_invalid_expression(self, calc):
        result = await calc.execute({"expression": "import os"})
        assert "error" in result


class TestTextProcessorSkill:
    @pytest.fixture
    def processor(self):
        return TextProcessorSkill()

    @pytest.mark.asyncio
    async def test_count_words(self, processor):
        result = await processor.execute({"operation": "count_words", "text": "hello world foo"})
        assert result["result"] == 3

    @pytest.mark.asyncio
    async def test_to_upper(self, processor):
        result = await processor.execute({"operation": "to_upper", "text": "hello"})
        assert result["result"] == "HELLO"

    @pytest.mark.asyncio
    async def test_to_lower(self, processor):
        result = await processor.execute({"operation": "to_lower", "text": "HELLO"})
        assert result["result"] == "hello"

    @pytest.mark.asyncio
    async def test_reverse(self, processor):
        result = await processor.execute({"operation": "reverse", "text": "abc"})
        assert result["result"] == "cba"

    @pytest.mark.asyncio
    async def test_unknown_operation(self, processor):
        result = await processor.execute({"operation": "unknown", "text": "abc"})
        assert "error" in result


class TestSkillManager:
    def test_register_and_get(self):
        manager = SkillManager()
        tool = CalculatorSkill()
        manager.register(tool)
        assert manager.get("calculator") is tool

    def test_get_nonexistent(self):
        manager = SkillManager()
        assert manager.get("nonexistent") is None

    def test_list_skills(self):
        registry = create_default_registry()
        tools = registry.list_skills()
        assert len(tools) == 6
        names = [t["name"] for t in tools]
        assert "calculator" in names
        assert "text_processor" in names
        assert "read_file" in names
        assert "text_processor" in names

    def test_skill_names(self):
        registry = create_default_registry()
        assert "calculator" in registry.skill_names
        assert "execute_command" in registry.skill_names
