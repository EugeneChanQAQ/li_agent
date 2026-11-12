import unittest

from pydantic import ValidationError

from harness.provider.models import ToolCall


class TestToolCall(unittest.TestCase):
    def test_missing_input_defaults_to_empty_dict(self):
        """省略工具参数时，默认得到空字典。"""
        call = ToolCall(
            tool_call_id="call_001",
            tool_name="get_current_time",
        )

        self.assertEqual(call.tool_input, {})

    def test_missing_call_id_is_rejected(self):
        """缺少调用 ID 时，应当校验失败。"""
        with self.assertRaises(ValidationError) as caught:
            ToolCall(tool_name="get_current_time")

        errors = caught.exception.errors()
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["loc"], ("tool_call_id",))
        self.assertEqual(errors[0]["type"], "missing")

    def test_default_inputs_are_independent(self):
        """修改一个调用的默认参数，不影响另一个调用。"""
        first = ToolCall(
            tool_call_id="call_001",
            tool_name="search_news",
        )
        second = ToolCall(
            tool_call_id="call_002",
            tool_name="search_news",
        )

        first.tool_input["query"] = "半导体"

        self.assertEqual(second.tool_input, {})


if __name__ == "__main__":
    unittest.main()
